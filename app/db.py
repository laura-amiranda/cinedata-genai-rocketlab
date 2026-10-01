"""Execução segura (somente leitura) de SQL contra o cinerocket.db.

Guardrails aplicados:
1. Só aceita comandos que comecem com SELECT ou WITH (CTE).
2. Bloqueia por palavra-chave qualquer comando de escrita/DDL/PRAGMA/ATTACH,
   mesmo que apareça dentro de uma CTE ou subquery.
3. Permite apenas uma única instrução (sem ";" encadeando múltiplos comandos).
4. Abre o arquivo SQLite em modo read-only (URI "mode=ro"), então mesmo que
   os filtros de texto falhem, o SO/SQLite recusa qualquer escrita física.
5. Limita o tempo de execução (via progress handler) e a quantidade de
   linhas retornadas, para evitar consultas que travem o processo ou
   devolvam uma tabela inteira para o LLM.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings

_FORBIDDEN_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "REPLACE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "REINDEX",
    "TRIGGER",
    "TRANSACTION",
    "COMMIT",
    "ROLLBACK",
)

_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


class UnsafeQueryError(ValueError):
    """Levantado quando a query proposta pelo agente não passa nos guardrails."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict]
    truncated: bool
    row_count: int
    elapsed_seconds: float = field(default=0.0)


def validate_sql_query(sql: str) -> str:
    """Valida que `sql` é uma única consulta SELECT/WITH somente leitura.

    Retorna o SQL "limpo" (sem ponto e vírgula final) se for seguro,
    ou levanta UnsafeQueryError com um motivo legível para o agente
    (para que ele possa tentar gerar uma query corrigida).
    """
    if not sql or not sql.strip():
        raise UnsafeQueryError("A query está vazia.")

    cleaned = sql.strip()

    # remove um único ";" final, se houver, mas recusa múltiplas instruções
    body = cleaned[:-1] if cleaned.endswith(";") else cleaned
    if ";" in body:
        raise UnsafeQueryError(
            "Apenas uma única instrução SQL é permitida por chamada (sem ';' no meio da query)."
        )

    if not _ALLOWED_START.match(body):
        raise UnsafeQueryError(
            "Só são permitidas consultas que comecem com SELECT ou WITH. "
            "Comandos de escrita/DDL não são permitidos neste banco."
        )

    upper = body.upper()
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            raise UnsafeQueryError(
                f"A palavra-chave '{keyword}' não é permitida. Use apenas SELECT/WITH de leitura."
            )

    return body


def run_query(sql: str, db_path: Path | None = None) -> QueryResult:
    """Executa uma query SELECT já validada e retorna até `settings.sql_max_rows` linhas."""
    safe_sql = validate_sql_query(sql)
    path = db_path or settings.db_path

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Banco de dados não encontrado em '{path}'. Baixe o cinerocket.db (veja o README) "
            "e configure DB_PATH no .env se necessário."
        )

    uri = f"file:{Path(path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=settings.sql_timeout_seconds)
    conn.row_factory = sqlite3.Row

    start = time.monotonic()
    deadline = start + settings.sql_timeout_seconds

    def _progress_handler() -> int:
        # retornar != 0 interrompe a query em andamento
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_progress_handler, 1000)

    try:
        cur = conn.execute(safe_sql)
        fetched = cur.fetchmany(settings.sql_max_rows + 1)
        columns = [d[0] for d in cur.description] if cur.description else []
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise TimeoutError(
                f"A consulta excedeu o limite de {settings.sql_timeout_seconds}s e foi interrompida. "
                "Tente uma consulta mais restrita (filtros, LIMIT)."
            ) from exc
        raise
    finally:
        conn.close()

    truncated = len(fetched) > settings.sql_max_rows
    rows = [dict(r) for r in fetched[: settings.sql_max_rows]]

    return QueryResult(
        columns=columns,
        rows=rows,
        truncated=truncated,
        row_count=len(rows),
        elapsed_seconds=time.monotonic() - start,
    )
