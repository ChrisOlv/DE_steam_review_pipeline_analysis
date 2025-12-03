import argparse
import logging
import time
import os
import requests
import pymssql
from dotenv import load_dotenv
from typing import Dict, Iterator, Optional, List

load_dotenv()  

BASE_URL = "https://store.steampowered.com/appreviews/{app_id}"

def connect_sql(max_retries: int = 3, retry_delay: float = 5.0) -> pymssql.Connection:
    """
    Connecte à SQL Server avec retry pour gérer les erreurs temporaires Azure.
    """
    password = os.getenv("SQL_SERVER_PASSWORD")
    if not password:
        raise RuntimeError("Définis SQL_SERVER_PASSWORD dans ton .env")
    server = os.getenv("SQL_SERVER_SERVER")
    if not server:
        raise RuntimeError("Définis SQL_SERVER_SERVER dans ton .env")
    user = os.getenv("SQL_SERVER_USER")
    if not user:
        raise RuntimeError("Définis SQL_SERVER_USER dans ton .env")
    database = os.getenv("SQL_SCHEMA")
    if not database:
        raise RuntimeError("Définis SQL_SCHEMA dans ton .env")
    
    for attempt in range(1, max_retries + 1):
        try:
            return pymssql.connect(
                server=server,
                user=user,
                password=password,
                database=database,
                port=1433,
                tds_version="7.4"
            )
        except Exception as e:
            error_code = None
            if hasattr(e, 'args') and len(e.args) > 0:
                error_code = e.args[0]
            
            # Erreur 40613 = Database not currently available (Azure SQL)
            # On retry pour toutes les erreurs de connexion si ce n'est pas la dernière tentative
            is_retryable = (error_code == 40613 or 
                          "connection" in str(e).lower() or 
                          "not currently available" in str(e))
            
            if is_retryable and attempt < max_retries:
                logging.warning(f"Tentative de connexion {attempt}/{max_retries} échouée: {e}")
                logging.info(f"Réessai dans {retry_delay} secondes...")
                time.sleep(retry_delay)
                continue
            # Si c'est la dernière tentative ou erreur non-retryable, on relance
            raise
    raise RuntimeError(f"Impossible de se connecter après {max_retries} tentatives")




def init_schema(conn: pymssql.Connection):
    cursor = conn.cursor()
    cursor.execute(
        """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='raw_reviews' AND xtype='U')
            CREATE TABLE raw_reviews (
                recommendationid NVARCHAR(60) PRIMARY KEY,
                author_steamid NVARCHAR(30),
                language NVARCHAR(10),
                review NVARCHAR(MAX),
                voted_up BIT,
                votes_up INT,
                votes_funny INT,
                weighted_vote_score FLOAT,
                comment_count INT,
                steam_purchase BIT,
                received_for_free BIT,
                timestamp_created BIGINT,
                timestamp_updated BIGINT,
                playtime_at_review BIGINT,
                playtime_forever BIGINT,
                author_num_games_owned INT,
                author_num_reviews INT,
                author_playtime_forever BIGINT,
                author_playtime_last_two_weeks BIGINT,
                author_last_played BIGINT
            );
        """
    )
    cursor.execute(
        """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ingest_state' AND xtype='U')
            CREATE TABLE ingest_state (
                source NVARCHAR(100) PRIMARY KEY,
                last_review_ts BIGINT,
                last_run_at DATETIME2 DEFAULT SYSUTCDATETIME()
            );
        """
    )
    conn.commit()


def get_last_ingest_ts(conn: pymssql.Connection, source: str) -> Optional[int]:
    cursor = conn.cursor()
    cursor.execute("SELECT last_review_ts FROM ingest_state WHERE source = %s", (source,))
    row = cursor.fetchone()
    return row[0] if row else None


def update_ingest_state(conn: pymssql.Connection, source: str, last_ts: int):
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM ingest_state WHERE source = %s", (source,))
    if cursor.fetchone():
        cursor.execute("UPDATE ingest_state SET last_review_ts = %s, last_run_at = SYSUTCDATETIME() WHERE source = %s", (last_ts, source))
    else:
        cursor.execute("INSERT INTO ingest_state (source, last_review_ts, last_run_at) VALUES (%s, %s, SYSUTCDATETIME())", (source, last_ts))
    conn.commit()


def fetch_reviews_paginated(
    app_id: int,
    since_ts: Optional[int] = None,
    language: str = "all",
    review_type: str = "all",
    purchase_type: str = "all",
    num_per_page: int = 100,
    max_pages: Optional[int] = None,
    timeout: int = 20,
    sleep_between_pages: float = 0.5,
) -> Iterator[Dict]:
    """
    Yields raw review dicts from Steam reviews API, handling pagination via cursor.
    If since_ts is provided, yields only reviews with timestamp_created > since_ts and stops early when pages are mostly older.
    """
    cursor = "*"
    pages = 0
    session = requests.Session()
    session.headers.update({"User-Agent": "steam-reviews-ingest/1.0"})

    while True:
        params = {
            "json": 1,
            "filter": "recent",  # recent order for incremental
            "language": language,
            "review_type": review_type,
            "purchase_type": purchase_type,
            "cursor": cursor,
            "num_per_page": num_per_page,
        }
        url = BASE_URL.format(app_id=app_id)
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Steam API request failed: {e}")
            time.sleep(2)
            continue

        data = r.json()
        reviews = data.get("reviews", []) or []
        if not reviews:
            logging.info("No more reviews returned.")
            break

        older_count = 0
        for rv in reviews:
            ts = int(rv.get("timestamp_created") or 0)
            if since_ts is not None and ts <= since_ts:
                older_count += 1
                continue
            yield rv

        cursor = data.get("cursor", "")
        pages += 1

        if max_pages is not None and pages >= max_pages:
            logging.info(f"Reached max_pages={max_pages}; stopping pagination.")
            break

        # Early stop: page is mostly older than since_ts
        if since_ts is not None and older_count >= max(1, int(len(reviews) * 0.8)):
            logging.info("Reached mostly older reviews; stopping pagination.")
            break

        if not cursor:
            logging.info("No cursor provided; pagination ended.")
            break

        time.sleep(sleep_between_pages)

def normalize_review(rv: Dict) -> Dict:
    author = rv.get("author", {}) or {}
    return {
        "recommendationid": str(rv.get("recommendationid")),
        "author_steamid": str(author.get("steamid")) if author.get("steamid") else None,
        "language": rv.get("language"),
        "review": rv.get("review"),
        "voted_up": bool(rv.get("voted_up")),
        "votes_up": int(rv.get("votes_up") or 0),
        "votes_funny": int(rv.get("votes_funny") or 0),
        "weighted_vote_score": float(rv.get("weighted_vote_score") or 0.0),
        "comment_count": int(rv.get("comment_count") or 0),
        "steam_purchase": bool(rv.get("steam_purchase")),
        "received_for_free": bool(rv.get("received_for_free")),
        "timestamp_created": int(rv.get("timestamp_created") or 0),
        "timestamp_updated": int(rv.get("timestamp_updated") or 0),
        "playtime_at_review": int(rv.get("playtime_at_review") or 0),
        "playtime_forever": int(rv.get("playtime_forever") or 0),
        "author_num_games_owned": int(author.get("num_games_owned") or 0),
        "author_num_reviews": int(author.get("num_reviews") or 0),
        "author_playtime_forever": int(author.get("playtime_forever") or 0),
        "author_playtime_last_two_weeks": int(author.get("playtime_last_two_weeks") or 0),
        "author_last_played": int(author.get("last_played") or 0),
    }

UPSERT_SQL = """
IF NOT EXISTS (SELECT 1 FROM raw_reviews WHERE recommendationid = %s)
BEGIN
    INSERT INTO raw_reviews (
        recommendationid, author_steamid, language, review,
        voted_up, votes_up, votes_funny, weighted_vote_score,
        comment_count, steam_purchase, received_for_free,
        timestamp_created, timestamp_updated,
        playtime_at_review, playtime_forever,
        author_num_games_owned, author_num_reviews,
        author_playtime_forever, author_playtime_last_two_weeks,
        author_last_played
    )
    VALUES (
        %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
        %s
    )
END
"""

def upsert_batch(conn: pymssql.Connection, batch: List[Dict]) -> int:
    cursor = conn.cursor()
    count = 0
    for r in batch:
        params = (
            r.get("recommendationid"),  # Pour le IF NOT EXISTS
            r.get("recommendationid"), r.get("author_steamid"), r.get("language"), r.get("review"),
            r.get("voted_up"), r.get("votes_up"), r.get("votes_funny"), r.get("weighted_vote_score"),
            r.get("comment_count"), r.get("steam_purchase"), r.get("received_for_free"),
            r.get("timestamp_created"), r.get("timestamp_updated"),
            r.get("playtime_at_review"), r.get("playtime_forever"),
            r.get("author_num_games_owned"), r.get("author_num_reviews"),
            r.get("author_playtime_forever"), r.get("author_playtime_last_two_weeks"),
            r.get("author_last_played")
        )
        cursor.execute(UPSERT_SQL, params)
        count += 1
    conn.commit()
    return count


def main():
    parser = argparse.ArgumentParser(description="Incremental ingest Steam reviews into Azure SQL.")
    parser.add_argument("--app-id", type=int, default=3697560, help="Steam App ID (default: 3697560)")
    parser.add_argument("--language", type=str, default="all", help="Review language filter (default: all)")
    parser.add_argument("--review-type", type=str, default="all", help="Review type filter (all/positive/negative)")
    parser.add_argument("--purchase-type", type=str, default="all", help="Purchase type filter (all/steam/non_steam)")
    parser.add_argument("--num-per-page", type=int, default=100, help="Number of reviews per page (max 100)")
    parser.add_argument("--max-pages", type=int, default=5, help="Max pages to fetch (default: 5)")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between pages in seconds")
    
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    conn = connect_sql()
    init_schema(conn)

    source = f"steam_{args.app_id}"
    last_ts = get_last_ingest_ts(conn, source)
    logging.info(f"Last ingest ts for {source}: {last_ts}")

    batch: List[Dict] = []
    newest_ts = last_ts or 0
    total_raw = 0

    for rv in fetch_reviews_paginated(
        app_id=args.app_id,
        since_ts=last_ts,
        language=args.language,
        review_type=args.review_type,
        purchase_type=args.purchase_type,
        num_per_page=args.num_per_page,
        max_pages=args.max_pages,
        sleep_between_pages=args.sleep,
    ):
        total_raw += 1
        nr = normalize_review(rv)
        batch.append(nr)
        if nr["timestamp_created"] > newest_ts:
            newest_ts = nr["timestamp_created"]
        if len(batch) >= 500:
            upsert_batch(conn, batch)
            logging.info(f"Inserted batch of {len(batch)} reviews.")
            batch = []

    if batch:
        upsert_batch(conn, batch)
        logging.info(f"Inserted final batch of {len(batch)} reviews.")

    if newest_ts and newest_ts != (last_ts or 0):
        update_ingest_state(conn, source, newest_ts)
        logging.info(f"Updated ingest_state to {newest_ts}")

    print(f"Fetched {total_raw} new reviews; newest_ts={newest_ts}")

if __name__ == "__main__":
    main()