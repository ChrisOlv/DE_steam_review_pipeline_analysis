import os
import argparse
import logging
from datetime import datetime

import duckdb
from dotenv import load_dotenv

load_dotenv()


def connect_motherduck(db_name: str) -> duckdb.DuckDBPyConnection:
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN is not set in environment or .env")
    # Sanitize token and set in env for DuckDB MotherDuck connector
    token = token.strip()
    os.environ["MOTHERDUCK_TOKEN"] = token
    conn = duckdb.connect("md:")
    conn.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    conn.execute(f"USE {db_name}")
    return conn


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_table_to_parquet(conn: duckdb.DuckDBPyConnection, table_name: str, out_path: str) -> int:
    # Remove existing file to avoid any overwrite issues
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception as e:
        logging.warning(f"Could not remove existing file {out_path}: {e}")
    try:
        # Count rows first for reporting
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        # Export to parquet
        conn.execute("COPY (SELECT * FROM " + table_name + ") TO ? (FORMAT 'parquet')", [out_path])
        logging.info(f"Exported {row_count} rows from {table_name} to {out_path}")
        return row_count
    except Exception as e:
        logging.warning(f"Skipping export for {table_name}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Export key tables to parquet files for dataviz.")
    parser.add_argument("--db-name", default=os.getenv("MD_DB_NAME", "steam_analytics"), help="MotherDuck DB name")
    parser.add_argument("--out-dir", default="dataviz", help="Output folder for parquet files")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    logging.info(f"Starting export at {datetime.utcnow().isoformat()}Z")
    ensure_dir(args.out_dir)

    conn = connect_motherduck(args.db_name)

    total_exported = 0
    summary = []
    exports = {
        "ingest_state": os.path.join(args.out_dir, "ingest_state.parquet"),
        "llm_enrichment": os.path.join(args.out_dir, "llm_enrichment.parquet"),
        "raw_reviews": os.path.join(args.out_dir, "raw_reviews.parquet"),
    }

    for table, path in exports.items():
        rows = export_table_to_parquet(conn, table, path)
        total_exported += rows
        summary.append((table, rows, path))

    logging.info("Export summary:")
    for table, rows, path in summary:
        logging.info(f"- {table}: {rows} rows -> {path}")

    print(f"Done. Exported {total_exported} total rows across {len(summary)} tables.")


if __name__ == "__main__":
    main()
