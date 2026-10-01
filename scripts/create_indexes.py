"""Cria índices de performance no cinerocket.db (rodar uma única vez).

Alguns JOINs/agregações do agente (ex: "ator com mais filmes nos últimos N
anos") demoravam mais que o timeout de 10s porque faltavam índices em
colunas usadas em WHERE/JOIN (principalmente dim_people.tipo_pessoa, que
não tinha índice e é filtrada em quase toda pergunta sobre elenco/equipe).

Isso NÃO altera nenhum dado — só cria estruturas auxiliares de busca que
deixam as mesmas consultas mais rápidas. É seguro rodar mais de uma vez
(usa CREATE INDEX IF NOT EXISTS) e seguro rodar com o cinerocket.db aberto
em outro terminal (SQLite lida bem com isso).

Uso:
    python scripts/create_indexes.py
    python scripts/create_indexes.py --db outro_caminho.db
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

INDEXES = [
    ("idx_dim_people_tipo_pessoa", "dim_people", "tipo_pessoa"),
    ("idx_bridge_movie_genre_genre", "bridge_movie_genre", "sk_genre_id"),
    ("idx_bridge_movie_company_company", "bridge_movie_company", "sk_company_id"),
    ("idx_dim_reviews_movie", "dim_reviews", "sk_movie_id"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="cinerocket.db", help="Caminho do cinerocket.db")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Banco não encontrado em '{db_path}'. Rode a partir da raiz do projeto.")

    conn = sqlite3.connect(db_path)
    try:
        for index_name, table, column in INDEXES:
            start = time.monotonic()
            conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column})")
            conn.commit()
            print(f"OK  {index_name} ON {table}({column})  ({time.monotonic() - start:.2f}s)")
    finally:
        conn.close()

    print("\nÍndices criados/confirmados com sucesso.")


if __name__ == "__main__":
    main()
