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

SCHEMA_DESCRIPTION = """
ESQUEMA DO BANCO (cinerocket.db, modelo dimensional / estrela):

dim_movies (95.645 filmes) — chave: sk_movie_id (texto, hash)
  titulo, titulo_original, ano_lancamento (2016–2029), idioma_original,
  status_filme (ex: 'Released'), duracao_minutos, sinopse, url_poster,
  nota_tmdb, qtd_votos_tmdb, popularidade_tmdb, nota_imdb, qtd_votos_imdb

fact_movies_performance (95.645 linhas, 1:1 com dim_movies) — chave: sk_movie_id
  orcamento_usd, receita_usd, orcamento_brl, receita_brl, lucro_usd, lucro_brl
  ATENÇÃO: dados financeiros são ESPARSOS. Só ~3.370 filmes têm receita_brl
  preenchida e só ~1.630 têm orçamento E receita preenchidos. SEMPRE filtre
  com "WHERE receita_brl IS NOT NULL" (e/ou orcamento_brl) em perguntas sobre
  bilheteria/lucro/margem.

dim_genres (19 gêneros) — sk_genre_id, nome_genero (em inglês: Action, Drama, Comedy, ...)
bridge_movie_genre — sk_movie_id, sk_genre_id (N:N entre filmes e gêneros)

dim_companies (45.941 produtoras) — sk_company_id, nome_empresa
bridge_movie_company — sk_movie_id, sk_company_id (N:N entre filmes e produtoras)

dim_people (424.656 pessoas) — sk_person_id, nome_pessoa, tipo_pessoa
  (tipo_pessoa é um dos três valores: 'Diretor', 'Ator', 'Roteirista')
bridge_movie_person — sk_movie_id, sk_person_id, tipo_pessoa (papel da pessoa
  naquele filme especificamente; uma pessoa pode ter mais de um papel/filme)

dim_reviews (40.267 filmes com pelo menos 1 avaliação) — sk_movie_id,
  qtd_avaliacoes_usuarios, nota_media_usuarios (0 a 10)
  Já traz o agregado por filme (contagem + média). Prefira usá-la para
  perguntas de "nota média" / "mais avaliados".

movie_reviews (43.666 avaliações individuais) — id, sk_movie_review_id,
  sk_movie_id, name (nome de quem avaliou), rating (0 a 10), text, created_at

Todas as tabelas de fato/bridge se relacionam a dim_movies por sk_movie_id.
Nomes de tabelas e colunas são exatamente como escrito acima.
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
