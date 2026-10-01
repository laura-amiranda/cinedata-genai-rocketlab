# CineData Analytics — Agente Text-to-SQL

Agente que responde perguntas em linguagem natural sobre o catálogo de filmes
`cinerocket.db`, gerando e executando SQL (somente leitura) sobre o banco e
devolvendo a resposta em texto. Construído com [PydanticAI](https://ai.pydantic.dev/)
e modelos gratuitos da [OpenRouter](https://openrouter.ai/), exposto como um
módulo FastAPI (`POST /ask`).

> Atividade GenAI — Rocket Lab 2026.2

## Arquitetura

```
app/
  config.py   # configurações (.env): chave OpenRouter, modelo, caminho do banco
  db.py       # execução segura de SQL: valida SELECT/WITH, bloqueia escrita,
              # abre o SQLite em modo read-only, limita linhas e tempo
  agent.py    # dois agentes PydanticAI (sql_agent e answer_agent) + a função
              # ask(), que orquestra o pipeline de 2 chamadas (ver abaixo)
  schemas.py  # modelos Pydantic da API (AskRequest/AskResponse)
  main.py     # app FastAPI (endpoints /ask e /health)
tests/
  test_db_guardrails.py  # testes do módulo db.py (sem chamar o OpenRouter)
```

Fluxo de uma pergunta: `POST /ask` → `app.agent.ask()`, que roda um pipeline
de 2 chamadas ao LLM controlado pelo nosso próprio código Python (em vez de
"tool calling" automático — ver a nota no topo de `app/agent.py` sobre por
que essa escolha foi feita):

1. `sql_agent` recebe a pergunta + o schema do banco e devolve só o SQL
   (como um JSON `{"sql": "..."}`, via `PromptedOutput`).
2. Executamos esse SQL em Python com os guardrails de `app/db.py`. Se der
   erro, mandamos o erro de volta pro `sql_agent` (mesma conversa) pra ele
   corrigir, até um limite de tentativas.
3. `answer_agent` recebe a pergunta original + os dados retornados e devolve
   a resposta final em português.

A API retorna a resposta final junto com a lista de SQLs tentados, para
transparência/debug.

## 1. Pré-requisitos

- Python 3.11+
- Uma chave de API da OpenRouter (grátis): crie em https://openrouter.ai/keys
- O arquivo `cinerocket.db` (fornecido pelo professor/atividade)

## 2. Baixar o banco de dados

**O arquivo `cinerocket.db` (≈580 MB) não está versionado neste repositório**
porque excede o limite de 100 MB do GitHub (está no `.gitignore`).

1. Baixe o arquivo a partir do link compartilhado na atividade.
2. Coloque o arquivo na raiz do projeto com o nome `cinerocket.db`
   (ou em outro caminho de sua escolha, configurando `DB_PATH` no `.env`).

## 3. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edite o .env e cole sua OPENROUTER_API_KEY
```

## 4. Rodar os testes (não consome a cota da OpenRouter)

Os testes em `tests/test_db_guardrails.py` validam os guardrails de SQL
(bloqueio de INSERT/UPDATE/DELETE/DROP/PRAGMA/ATTACH, múltiplas instruções,
modo read-only) e a execução de queries reais contra o `cinerocket.db`.
**Não fazem nenhuma chamada ao LLM**, então podem ser rodados quantas vezes
quiser sem gastar a cota diária.

```bash
pytest -v
```

## 5. Rodar a API

```bash
uvicorn app.main:app --reload
```

Acesse a documentação interativa em http://localhost:8000/docs.

### Exemplo de uso

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Quais são os 5 filmes com maior receita, considerando apenas os que têm receita informada?"}'
```

Resposta:

```json
{
  "pergunta": "Quais são os 5 filmes com maior receita, considerando apenas os que têm receita informada?",
  "resposta": "Os 5 filmes com maior receita informada são: ...",
  "sql_executado": [
    "SELECT m.titulo, f.receita_brl FROM dim_movies m JOIN fact_movies_performance f ON f.sk_movie_id = m.sk_movie_id WHERE f.receita_brl IS NOT NULL ORDER BY f.receita_brl DESC LIMIT 5"
  ]
}
```

## 6. Sobre a cota gratuita da OpenRouter

O modelo padrão (`poolside/laguna-s-2.1:free`, configurável via
`OPENROUTER_MODEL` no `.env`) é gratuito, mas o tier free da OpenRouter
permite **apenas 50 requisições por dia** (e 20/min), e cada pergunta feita
ao agente normalmente gera 2 chamadas ao modelo (uma para gerar o SQL, outra
para formular a resposta final a partir dos dados), podendo gerar mais se
precisar corrigir uma query com erro. Por isso:

**Nota sobre a escolha do modelo:** os modelos `:free` da OpenRouter mudam de
tempos em tempos — alguns saem do catálogo gratuito (passam a exigir a versão
paga) e outros aparecem. Durante o desenvolvimento, testamos 3 modelos
diferentes: `nvidia/nemotron-3.5-lightning:free` e `z-ai/glm-5.2:free`
(sugeridos no material da disciplina) pararam de funcionar bem — o primeiro
raramente seguia o formato de saída esperado, o segundo saiu do tier
gratuito. O `nvidia/nemotron-3-ultra-550b-a55b:free`, apesar de mais
"inteligente", é um modelo muito grande (550B parâmetros) e teve fila de
vários minutos no tier gratuito. `poolside/laguna-s-2.1:free` (8B parâmetros
ativos) se mostrou um bom equilíbrio: rápido e confiável para gerar SQL. Se
ele parar de funcionar no futuro, confira a lista atual de modelos grátis em
https://openrouter.ai/models?q=:free — modelos menores tendem a responder
mais rápido no tier gratuito.

- Os testes automatizados (`pytest`) não chamam o LLM, então podem ser usados
  livremente para validar os guardrails sem gastar cota.
- Ao testar o agente manualmente via `/ask`, prefira poucas perguntas bem
  escolhidas (uma de cada categoria do enunciado) em vez de várias tentativas
  seguidas.
- Você pode checar sua cota restante em `GET https://openrouter.ai/api/v1/key`
  (use sua `OPENROUTER_API_KEY` no header `Authorization: Bearer ...`).
- A cota reseta à meia-noite UTC (21h em Fortaleza/BRT).

## 7. Guardrails de segurança do SQL

A função `run_query` (em `app/db.py`), chamada pelo nosso código após o
`sql_agent` gerar a consulta, só executa a query se, em conjunto:

1. Começar com `SELECT` ou `WITH` (recusa qualquer outra coisa);
2. Não conter nenhuma palavra-chave de escrita/DDL/controle de transação em
   nenhum ponto da query (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`,
   `CREATE`, `ATTACH`, `PRAGMA`, `COMMIT`, etc. — mesmo dentro de uma CTE);
3. For uma única instrução (sem `;` encadeando comandos);
4. For executada numa conexão SQLite aberta em **modo read-only**
   (`file:...?mode=ro`), como defesa em profundidade caso os filtros de texto
   acima falhem;
5. Respeitar um limite de linhas retornadas (200, configurável) e um timeout
   de execução (10s, configurável), para não travar o processo nem devolver
   uma tabela inteira para o LLM.

Se o SQL gerado pelo `sql_agent` violar alguma dessas regras (ou falhar por
outro motivo, como timeout), o erro é devolvido pro modelo na mesma conversa
e ele tem a chance de corrigir e tentar de novo, até `MAX_SQL_ATTEMPTS`
tentativas (3, em `app/agent.py`).

## 8. Schema do banco (`cinerocket.db`)

Modelo dimensional (estrela), todas as tabelas de fato/ponte se relacionam a
`dim_movies` por `sk_movie_id`:

| Tabela | Linhas | Descrição |
|---|---|---|
| `dim_movies` | 95.645 | Dados dos filmes: título, ano, duração, sinopse, notas TMDB/IMDB |
| `fact_movies_performance` | 95.645 | Orçamento e receita (USD/BRL) e lucro — **esparso**: só ~3.370 filmes têm receita e ~1.630 têm orçamento+receita preenchidos |
| `dim_genres` | 19 | Gêneros (em inglês) |
| `bridge_movie_genre` | — | N:N filme↔gênero |
| `dim_companies` | 45.941 | Produtoras |
| `bridge_movie_company` | — | N:N filme↔produtora |
| `dim_people` | 424.656 | Pessoas (diretores, atores, roteiristas) |
| `bridge_movie_person` | — | N:N filme↔pessoa, com o papel (`tipo_pessoa`) naquele filme |
| `dim_reviews` | 40.267 | Agregado por filme: quantidade e nota média das avaliações de usuários |
| `movie_reviews` | 43.666 | Avaliações individuais (nome, nota, texto, data) |

O system prompt completo do agente (em `app/agent.py`) descreve cada coluna
em detalhe e inclui o aviso sobre a esparsidade dos dados financeiros, para
que o agente sempre filtre `IS NOT NULL` em perguntas de bilheteria/lucro.

## 9. Limitações conhecidas / possíveis melhorias futuras

- Sem memória de conversa entre perguntas (cada `/ask` é uma run independente).
- Sem cache de respostas (cada pergunta sempre consulta o modelo).
- Sem fallback automático entre modelos gratuitos caso um esteja indisponível.
- Sem busca semântica sobre sinopses (só Text-to-SQL estruturado).
