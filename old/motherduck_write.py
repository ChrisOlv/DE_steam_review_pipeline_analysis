import os
import json
import argparse
import logging
from typing import Dict, List

import duckdb
from dotenv import load_dotenv

load_dotenv()  # charge .env s'il est présent (MOTHERDUCK_TOKEN, etc.)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_reviews (
    recommendationid TEXT,
    author_steamid TEXT,
    language TEXT,
    review TEXT,
    voted_up BOOLEAN,
    votes_up INTEGER,
    votes_funny INTEGER,
    weighted_vote_score DOUBLE,
    comment_count INTEGER,
    steam_purchase BOOLEAN,
    received_for_free BOOLEAN,
    timestamp_created BIGINT,
    timestamp_updated BIGINT,
    playtime_at_review BIGINT,
    playtime_forever BIGINT,
    author_num_games_owned INTEGER,
    author_num_reviews INTEGER,
    author_playtime_forever BIGINT,
    author_playtime_last_two_weeks BIGINT,
    author_last_played BIGINT
);
"""

UPSERT_SQL = """
INSERT INTO raw_reviews
SELECT
    ? AS recommendationid, ? AS author_steamid, ? AS language, ? AS review,
    ? AS voted_up, ? AS votes_up, ? AS votes_funny, ? AS weighted_vote_score,
    ? AS comment_count, ? AS steam_purchase, ? AS received_for_free,
    ? AS timestamp_created, ? AS timestamp_updated,
    ? AS playtime_at_review, ? AS playtime_forever,
    ? AS author_num_games_owned, ? AS author_num_reviews,
    ? AS author_playtime_forever, ? AS author_playtime_last_two_weeks,
    ? AS author_last_played
WHERE NOT EXISTS (
    SELECT 1 FROM raw_reviews WHERE recommendationid = ?
);
"""

def connect_motherduck(db_name: str) -> duckdb.DuckDBPyConnection:
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN is not set in the environment or .env")

    # 1) Connect to MotherDuck with token in the URI to bypass SSO
    conn = duckdb.connect(f"md:?motherduck_token={token}")

    # 2) Create the database if it doesn't exist and switch to it
    conn.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    conn.execute(f"USE {db_name}")
    return conn

def init_schema(conn: duckdb.DuckDBPyConnection):
    conn.execute(SCHEMA_SQL)

def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def upsert_batch(conn: duckdb.DuckDBPyConnection, batch: List[Dict]) -> int:
    count = 0
    for r in batch:
        params = [
            r.get("recommendationid"), r.get("author_steamid"), r.get("language"), r.get("review"),
            r.get("voted_up"), r.get("votes_up"), r.get("votes_funny"), r.get("weighted_vote_score"),
            r.get("comment_count"), r.get("steam_purchase"), r.get("received_for_free"),
            r.get("timestamp_created"), r.get("timestamp_updated"),
            r.get("playtime_at_review"), r.get("playtime_forever"),
            r.get("author_num_games_owned"), r.get("author_num_reviews"),
            r.get("author_playtime_forever"), r.get("author_playtime_last_two_weeks"),
            r.get("author_last_played"),
            r.get("recommendationid"),
        ]
        conn.execute(UPSERT_SQL, params)
        count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description="Write normalized Steam reviews JSONL into MotherDuck.")
    parser.add_argument("--in-jsonl", required=True, help="Path to normalized reviews JSONL (from steam_ingest.py)")
    parser.add_argument("--db-name", default=os.getenv("MD_DB_NAME", "steam_analytics"), help="MotherDuck DB name")
    parser.add_argument("--batch-size", type=int, default=500, help="Insert batch size")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    conn = connect_motherduck(args.db_name)
    init_schema(conn)

    rows = read_jsonl(args.in_jsonl)
    total_inserted = 0

        # Process and insert batches
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        total_inserted += upsert_batch(conn, batch)
        logging.info(f"Inserted batch of {len(batch)} rows (total so far: {total_inserted}).")
    count = conn.execute("SELECT COUNT(*) FROM raw_reviews").fetchone()[0]
    print(f"Done. Inserted (attempted) {total_inserted} rows. raw_reviews now has {count} rows.")


if __name__ == "__main__":
    main()