"""Agente Text-to-SQL do CineData Analytics, usando PydanticAI + OpenRouter.

IMPORTANTE sobre a arquitetura: a primeira versão deste agente usava
"tool calling" nativo (o LLM decide sozinho quando chamar uma função Python).
Na prática, os modelos gratuitos (:free) da OpenRouter testados aqui não
lidam bem com isso — ou simplesmente não chamam a ferramenta e devolvem o
próprio raciocínio como resposta, ou retornam texto corrompido.

Por isso o fluxo foi trocado para um pipeline de 2 chamadas, controlado pelo
nosso próprio código Python em vez de depender da API de function-calling:

  1) "sql_agent": recebe a pergunta + o schema do banco e devolve APENAS um
     JSON { "sql": "..." } (via PromptedOutput — o modelo só precisa gerar
     texto/JSON, não usar a feature de tools da API).
  2) Nós executamos esse SQL em Python, com os guardrails de app/db.py.
     Se der erro, mandamos o erro de volta pro sql_agent (mesma conversa,
     via message_history) pra ele tentar corrigir, até um limite de tentativas.
  3) "answer_agent": recebe a pergunta original + os dados retornados pela
     consulta e devolve a resposta final em português, em texto simples.

Esse desenho é bem mais robusto com modelos gratuitos fracos, porque cada
chamada ao LLM só precisa gerar texto (JSON ou prosa) — nunca depende do
modelo "decidir" invocar uma função.
"""

from __future__ import annotations

import json
import sqlite3

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import settings
from app.db import QueryResult, UnsafeQueryError, run_query

MAX_SQL_ATTEMPTS = 3

# Guardrail determinístico de INTENÇÃO, separado do guardrail de SQL do
# db.py. O db.py bloqueia a *query* se ela tentar escrever — mas se a
# pergunta pedir uma operação de escrita (ex: "apague os filmes de Horror"),
# o sql_agent pode gerar um SELECT "inofensivo" (ex: buscando os IDs que
# seriam apagados) e o answer_agent pode responder como se um próximo passo
# de escrita fosse possível, o que é enganoso: este agente nunca escreve no
# banco. Por isso, perguntas com intenção de escrita são recusadas aqui,
# antes de qualquer chamada ao LLM (também economiza cota da OpenRouter).
WRITE_INTENT_KEYWORDS = (
    "apag",  # apague, apagar, apagando
    "delet",  # deletar, delete
    "exclu",  # exclua, excluir, exclusão
    "remov",  # remova, remover, removido
    "atualiz",  # atualize, atualizar, atualização
    "insir",  # insira
    "inserir",
    "modific",  # modifique, modificar
    "alter",  # altere, alterar (cuidado: também casa "alternativa", aceitável aqui)
    "sobrescrev",  # sobrescrever, sobrescreva
    "drop",
    "truncat",  # truncate, truncar
    "update",
    "insert",
    "delete",
)

RECUSA_ESCRITA = (
    "Não posso fazer isso — este agente só executa consultas de leitura "
    "sobre o catálogo de filmes (SELECT), nunca inserção, atualização ou "
    "exclusão de dados. Se quiser, posso responder perguntas sobre esses "
    "filmes em vez de alterá-los (ex: quantos são, quais notas têm, etc.)."
)


def _tem_intencao_de_escrita(pergunta: str) -> bool:
    texto = pergunta.lower()
    return any(keyword in texto for keyword in WRITE_INTENT_KEYWORDS)

SCHEMA_DESCRIPTION = """
ESQUEMA DO BANCO (cinerocket.db, modelo dimensional / estrela). Os nomes de
tabela/coluna abaixo foram conferidos direto no banco (PRAGMA table_info) —
use EXATAMENTE esses nomes, não invente variações.

dim_movies (95.645 filmes) — chave: sk_movie_id (texto, hash)
  id_filme, titulo, data_lancamento, ano_lancamento, duracao_minutos,
  idioma_original (ATENÇÃO: sempre NULL no banco inteiro, não use em filtro),
  status_filme (valores possíveis: 'Lançado', 'Pós-Produção', 'Em Produção',
  'Planejado'), sinopse, url_poster, url_backdrop.
  NÃO existem nota/popularidade/votos nesta tabela — isso fica em
  fact_movies_performance (ver abaixo). NÃO existe coluna titulo_original.

fact_movies_performance (95.645 linhas, 1:1 com dim_movies) — chave: sk_movie_id
  orcamento_usd, receita_usd, lucro_usd, orcamento_brl, receita_brl, lucro_brl,
  popularidade, nota_tmdb, qtd_tmdb, nota_imdb, qtd_imdb.
  ATENÇÃO (dados esparsos, confira sempre com IS NOT NULL antes de usar):
  - orcamento_brl/usd: preenchido em ~8% dos filmes (~7.900 de 95.645).
  - receita_brl/usd: preenchido em ~3,5% dos filmes (~3.370 de 95.645).
  - lucro_brl/usd: CUIDADO — essa coluna vem preenchida (às vezes com 0) para
    quase todos os filmes MESMO quando orcamento/receita estão NULL. Nunca
    use lucro_brl/usd sozinho como indicador de "tem dado financeiro"; sempre
    filtre por receita_brl IS NOT NULL (e orcamento_brl IS NOT NULL quando a
    pergunta envolver margem/lucro) antes de calcular ou ordenar por lucro.
  - popularidade, nota_tmdb, qtd_tmdb, nota_imdb, qtd_imdb: a coluna em si é
    bem mais completa (85–100% não-NULL), mas CUIDADO: nota_tmdb usa 0 (não
    NULL) pra representar "sem avaliação" — ~36.000 filmes têm nota_tmdb = 0
    E qtd_tmdb = 0 ao mesmo tempo (ou seja, filme nunca avaliado, não é uma
    nota real de zero). Em perguntas de ranking/divergência/comparação
    envolvendo nota_tmdb, filtre também por qtd_tmdb > 0 (não só
    nota_tmdb IS NOT NULL), senão filmes "sem avaliação" poluem o resultado
    como se fossem notas baixas reais. nota_imdb não tem esse problema (é
    quase sempre uma nota real quando preenchida).

dim_genres (19 gêneros) — sk_genre_id, nome_genero (em inglês: Action, Drama, Comedy, ...)
bridge_movie_genre — sk_movie_id, sk_genre_id (N:N entre filmes e gêneros)

dim_companies (45.941 produtoras) — sk_company_id, nome_produtora
bridge_movie_company — sk_movie_id, sk_company_id (N:N entre filmes e produtoras)

dim_people (424.656 pessoas) — sk_person_id, nome_pessoa, tipo_pessoa
  (tipo_pessoa é um dos três valores: 'Diretor', 'Ator', 'Roteirista' — essa é
  a ÚNICA tabela que tem essa coluna de papel/função da pessoa)
bridge_movie_person — sk_movie_id, sk_person_id (N:N entre filmes e pessoas;
  NÃO tem coluna tipo_pessoa — pra saber o papel da pessoa, faça JOIN com
  dim_people e use dim_people.tipo_pessoa)

dim_reviews (40.267 filmes com pelo menos 1 avaliação) — sk_review_id,
  sk_movie_id, qtd_avaliacoes_usuarios, nota_media_usuarios (0 a 10)
  Já traz o agregado por filme (contagem + média, idêntico ao que se obtém
  agregando movie_reviews). Prefira usá-la para perguntas de "nota média" /
  "mais avaliados" em vez de agregar movie_reviews na mão. ATENÇÃO: o número
  de avaliações por filme é BAIXO (no máximo ~13 no banco inteiro) — não
  assuma que existem filmes com dezenas ou centenas de avaliações.

movie_reviews (43.666 avaliações individuais) — id, sk_movie_review_id,
  sk_movie_id, name (nome de quem avaliou), rating (0 a 10), text, created_at

Todas as tabelas de fato/bridge se relacionam a dim_movies por sk_movie_id.
Nomes de tabelas e colunas são exatamente como escrito acima — não existem
outras colunas além das listadas.
""".strip()

SQL_SYSTEM_PROMPT = f"""
Você converte perguntas em português sobre um catálogo de filmes em uma
única consulta SQL (SQLite) de leitura.

REGRAS:
- Gere APENAS SELECT ou WITH (CTE). Nunca INSERT/UPDATE/DELETE/DDL/PRAGMA.
- Uma única instrução SQL, sem ';' no meio.
- Em perguntas de "top N" / listagem, use LIMIT.
- Em perguntas de bilheteria/lucro/margem, filtre os campos financeiros
  relevantes com IS NOT NULL, pois são esparsos.
- CUIDADO com "margem de lucro MÉDIA" agrupada (ex: por gênero/produtora):
  o banco tem alguns orcamento_brl/usd com valores absurdamente baixos (ex:
  R$ 712 para um filme, claramente um erro de dado, não um orçamento real).
  Calcular AVG(lucro/orcamento) por filme e depois agrupar é MUITO sensível
  a esses outliers (um único filme com orçamento quase zero pode dominar a
  média inteira do grupo). Prefira SEMPRE a margem agregada:
  SUM(lucro_brl) / SUM(orcamento_brl) por grupo, em vez de
  AVG(lucro_brl / orcamento_brl) por filme — dá um resultado muito mais
  robusto e condizente com a realidade dos dados.
- O banco tem TRÊS "notas" diferentes e a pergunta pode não dizer qual quer:
  nota_imdb e nota_tmdb (em fact_movies_performance, notas "oficiais" do
  filme, bem mais completas) e nota_media_usuarios (em dim_reviews, nota
  interna calculada a partir de avaliações de usuários — é MUITO esparsa,
  poucos filmes têm mais que 2-3 avaliações, então é pouco confiável pra
  rankings). Se a pergunta falar só em "nota"/"nota média" sem especificar
  de quem (ex: "diretores com maior nota média"), use nota_imdb como padrão.
  Só use nota_media_usuarios quando a pergunta mencionar explicitamente
  "avaliação(ões) de usuários" ou "nota dos usuários".
- Use exatamente os nomes de tabela/coluna do schema abaixo.

{SCHEMA_DESCRIPTION}
""".strip()

ANSWER_SYSTEM_PROMPT = """
Você é o assistente do CineData Analytics. Você recebe uma pergunta do
usuário em português e o resultado (já executado) de uma consulta SQL sobre
o catálogo de filmes. Responda a pergunta em texto corrido, em português,
direto e claro, citando os números relevantes.

- Baseie-se SOMENTE nos dados fornecidos. Não invente números.
- Se a lista de linhas estiver vazia, diga que não foram encontrados dados
  para essa pergunta (não invente um resultado).
- Se os dados parecerem parciais (ex: poucos filmes com receita informada),
  mencione essa limitação na resposta.
- Se a pergunta pedir um superlativo ("qual É o maior/melhor/menor...") e os
  dados trouxerem exatamente 1 linha, isso JÁ É a resposta completa — a
  consulta SQL já ordenou e limitou o resultado pra trazer só o vencedor de
  propósito. NÃO diga que "só veio 1 registro, não dá pra comparar" ou que
  faltam dados pra fazer um ranking — isso é um erro de interpretação, a
  pergunta pedia só o topo mesmo, e 1 linha é o esperado. Responda
  diretamente qual é o vencedor e o valor.
- Se a pergunta mencionava "nota"/"nota média" de forma ambígua (sem dizer
  TMDB, IMDb ou usuários) e os dados vieram da nota do IMDb (padrão nesse
  caso), deixe isso claro na resposta (ex: "com base na nota do IMDb").
- Não repita o SQL nem fale sobre "a consulta" — responda como se estivesse
  conversando diretamente com a pessoa que perguntou.
""".strip()


class SQLPlan(BaseModel):
    """O plano de consulta: apenas a query SQL que responde à pergunta."""

    sql: str = Field(description="Uma única consulta SQL (SELECT/WITH) que responde à pergunta do usuário.")


def _build_model() -> OpenAIChatModel:
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY não configurada. Copie .env.example para .env e preencha sua chave "
            "(veja https://openrouter.ai/keys)."
        )
    return OpenAIChatModel(
        settings.openrouter_model,
        provider=OpenAIProvider(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        ),
    )


def build_sql_agent() -> Agent[None, SQLPlan]:
    model = _build_model()
    return Agent(
        model,
        output_type=PromptedOutput(SQLPlan),
        system_prompt=SQL_SYSTEM_PROMPT,
        retries=MAX_SQL_ATTEMPTS,
    )


def build_answer_agent() -> Agent[None, str]:
    model = _build_model()
    return Agent(
        model,
        output_type=str,
        system_prompt=ANSWER_SYSTEM_PROMPT,
    )


sql_agent = build_sql_agent()
answer_agent = build_answer_agent()


def ask(pergunta: str) -> tuple[str, list[str]]:
    """Executa o pipeline de ponta a ponta para uma pergunta e retorna
    (resposta_em_texto, lista_de_sqls_tentados)."""
    if _tem_intencao_de_escrita(pergunta):
        return RECUSA_ESCRITA, []

    tentativas: list[str] = []

    plan_result = sql_agent.run_sync(pergunta)
    sql = plan_result.output.sql
    tentativas.append(sql)

    result: QueryResult | None = None
    last_error: Exception | None = None

    for _ in range(MAX_SQL_ATTEMPTS):
        try:
            result = run_query(sql)
            break
        except (UnsafeQueryError, sqlite3.OperationalError, TimeoutError, FileNotFoundError) as exc:
            last_error = exc
            retry_result = sql_agent.run_sync(
                f"A consulta anterior falhou com este erro: {exc}\n"
                "Gere uma nova consulta SQL corrigida que responda à pergunta original.",
                message_history=plan_result.all_messages(),
            )
            plan_result = retry_result
            sql = plan_result.output.sql
            tentativas.append(sql)

    if result is None:
        raise RuntimeError(
            f"Não foi possível gerar uma consulta SQL válida após {MAX_SQL_ATTEMPTS} tentativas. "
            f"Último erro: {last_error}"
        )

    dados = {
        "colunas": result.columns,
        "linhas": result.rows,
        "quantidade_linhas": result.row_count,
        "truncado": result.truncated,
    }
    answer_prompt = (
        f"Pergunta do usuário: {pergunta}\n\n"
        f"Resultado da consulta SQL (JSON): {json.dumps(dados, ensure_ascii=False, default=str)}"
    )
    answer_result = answer_agent.run_sync(answer_prompt)

    return answer_result.output, tentativas
