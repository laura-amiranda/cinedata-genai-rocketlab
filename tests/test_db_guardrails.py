"""Testes dos guardrails de SQL. Não fazem nenhuma chamada ao OpenRouter,
então podem ser rodados livremente sem gastar a cota de 50 req/dia."""

from pathlib import Path

import pytest

from app.db import UnsafeQueryError, run_query, validate_sql_query

DB_PATH = Path(__file__).resolve().parent.parent / "cinerocket.db"
requires_db = pytest.mark.skipif(not DB_PATH.exists(), reason="cinerocket.db não encontrado (veja o README)")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM dim_movies LIMIT 5",
        "select titulo from dim_movies",
        "WITH t AS (SELECT * FROM dim_movies) SELECT * FROM t",
        "  SELECT 1  ",
    ],
)
def test_validate_sql_query_allows_select_and_with(sql):
    assert validate_sql_query(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE dim_movies",
        "DELETE FROM dim_movies",
        "UPDATE dim_movies SET titulo='x'",
        "INSERT INTO dim_movies VALUES (1)",
        "PRAGMA table_info(dim_movies)",
        "ATTACH DATABASE 'x.db' AS x",
        "SELECT * FROM dim_movies; DROP TABLE dim_movies",
        "",
        "   ",
        "UPDATE_LOG_TABLE_SELECT",  # não começa com SELECT/WITH
    ],
)
def test_validate_sql_query_blocks_unsafe(sql):
    with pytest.raises(UnsafeQueryError):
        validate_sql_query(sql)


def test_validate_sql_query_blocks_keyword_inside_cte():
    sql = "WITH t AS (DELETE FROM dim_movies) SELECT * FROM t"
    with pytest.raises(UnsafeQueryError):
        validate_sql_query(sql)


@requires_db
def test_run_query_returns_rows():
    result = run_query("SELECT sk_movie_id, titulo FROM dim_movies LIMIT 3")
    assert result.row_count == 3
    assert "titulo" in result.columns
    assert not result.truncated


@requires_db
def test_run_query_respects_row_limit():
    result = run_query("SELECT sk_movie_id FROM dim_movies")
    from app.config import settings

    assert result.row_count == settings.sql_max_rows
    assert result.truncated


@requires_db
def test_run_query_rejects_write_even_if_text_filter_missed():
    # defesa em profundidade: mesmo que a validação de texto falhasse,
    # a conexão é aberta em modo read-only
    with pytest.raises(UnsafeQueryError):
        run_query("DELETE FROM dim_movies")
