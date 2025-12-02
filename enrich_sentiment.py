import os
import json
import argparse
import logging
from datetime import datetime
from typing import Optional, List, Tuple

import duckdb
from dotenv import load_dotenv
from openai import AzureOpenAI
from prompts import get_sentiment_prompt, get_summary_prompt, get_keywords_and_themes_prompt, get_pros_and_cons_prompt, get_aspect_scores_prompt, get_feature_requests_prompt, get_language_detected_prompt, get_quote_highlight_prompt, get_toxicity_prompt, get_tone_quality_flags_prompt, get_bug_info_prompt, get_feature_request_bonus_prompt, get_nps_and_emotion_prompt, get_playtime_bucket_prompt, get_reviewer_experience_prompt, get_pertinence_prompt

load_dotenv()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_enrichment (
    recommendationid TEXT PRIMARY KEY,
    sentiment_label TEXT,
    sentiment_score DOUBLE,
    summary_10_words TEXT,
    tl_dr TEXT,
    keywords TEXT,
    themes TEXT,
    pros TEXT,
    cons TEXT,
    aspect_scores TEXT,
    feature_requests TEXT,
    language_detected TEXT,
    normalized_text_en TEXT,
    -- Bonus set
    quote_highlight TEXT,
    toxicity_score DOUBLE,
    sarcasm_flag BOOLEAN,
    humor_flag BOOLEAN,
    spam_flag BOOLEAN,
    coherence_score DOUBLE,
    bug_report BOOLEAN,
    bug_type TEXT,
    steps_hint TEXT,
    feature_request BOOLEAN,
    requested_features TEXT,
    suggestion_text TEXT,
    playtime_bucket TEXT,
    reviewer_experience_level TEXT,
    nps_category TEXT,
    emotion_primary TEXT,
    pertinence BOOLEAN
);
"""






def connect_motherduck(db_name: str) -> duckdb.DuckDBPyConnection:
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN is not set in environment or .env")
    token = token.strip()
    os.environ["MOTHERDUCK_TOKEN"] = token
    try:
        conn = duckdb.connect("md:")
        conn.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        conn.execute(f"USE {db_name}")
        return conn
    except Exception as e:
        logging.error(f"MotherDuck connection failed: {e}")
        raise


def get_azure_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    if not endpoint or not api_key:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

SYSTEM_PROMPT = (
    "You are a strict sentiment classifier for Steam game reviews. "
    "Return ONLY a JSON object with one key 'sentiment_label' whose value is one of: "
    "'positive', 'neutral', or 'negative'. Consider overall tone, not just single words."
)

def classify_sentiment(client: AzureOpenAI, deployment: str, text: str) -> Tuple[str, float]:
    try:
        prompt = get_sentiment_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a strict sentiment classifier for Steam game reviews."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        label = (parsed.get("sentiment_label") or "").strip().lower()
        score_raw = parsed.get("sentiment_score")
        try:
            score = float(score_raw) if score_raw is not None else 0.5
        except (TypeError, ValueError):
            score = 0.5
        if label not in ("positive", "neutral", "negative"):
            label = "neutral"
        score = max(0.0, min(1.0, score))
        return label, score
    except Exception as e:
        logging.warning(f"Sentiment classification failed: {e}")
        return "neutral", 0.5


def detect_language_and_normalize(client: AzureOpenAI, deployment: str, text: str) -> Tuple[str, str]:
    try:
        prompt = get_language_detected_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Detect language and translate to English; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        lang = (parsed.get("language_detected") or "").strip().lower() or "en"
        norm = parsed.get("normalized_text_en") or (text or "")
        return lang, norm
    except Exception as e:
        logging.warning(f"Language detection/normalization failed: {e}")
        return "en", (text or "")


def generate_summary(client: AzureOpenAI, deployment: str, text: str) -> Tuple[str, str]:
    try:
        prompt = get_summary_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Summarize review; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return parsed.get("summary_10_words") or "", parsed.get("tl_dr") or ""
    except Exception as e:
        logging.warning(f"Summary generation failed: {e}")
        return "", ""


def extract_keywords_themes(client: AzureOpenAI, deployment: str, text: str) -> Tuple[List[str], List[str]]:
    try:
        prompt = get_keywords_and_themes_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Extract keywords and themes; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        kw = parsed.get("keywords") or []
        th = parsed.get("themes") or []
        kw = [str(x).strip() for x in kw if str(x).strip()] if isinstance(kw, list) else []
        th = [str(x).strip() for x in th if str(x).strip()] if isinstance(th, list) else []
        return kw, th
    except Exception as e:
        logging.warning(f"Keywords/themes extraction failed: {e}")
        return [], []


def extract_pros_cons(client: AzureOpenAI, deployment: str, text: str) -> Tuple[List[str], List[str]]:
    try:
        prompt = get_pros_and_cons_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Extract pros and cons; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        pros = parsed.get("pros") or []
        cons = parsed.get("cons") or []
        pros = [str(x).strip() for x in pros if str(x).strip()] if isinstance(pros, list) else []
        cons = [str(x).strip() for x in cons if str(x).strip()] if isinstance(cons, list) else []
        return pros, cons
    except Exception as e:
        logging.warning(f"Pros/cons extraction failed: {e}")
        return [], []


def score_aspects(client: AzureOpenAI, deployment: str, text: str) -> dict:
    try:
        prompt = get_aspect_scores_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Score aspects; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return {}
        scores = {}
        for k, v in parsed.items():
            try:
                val = float(v)
                scores[k] = max(-1.0, min(1.0, val))
            except (TypeError, ValueError):
                continue
        return scores
    except Exception as e:
        logging.warning(f"Aspect scoring failed: {e}")
        return {}


def find_feature_requests(client: AzureOpenAI, deployment: str, text: str) -> List[str]:
    try:
        prompt = get_feature_requests_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Find feature requests; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        reqs = parsed.get("feature_requests") or []
        reqs = [str(x).strip() for x in reqs if str(x).strip()] if isinstance(reqs, list) else []
        return reqs
    except Exception as e:
        logging.warning(f"Feature requests extraction failed: {e}")
        return []


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in ("true", "yes", "1")


def generate_quote_highlight(client: AzureOpenAI, deployment: str, text: str) -> str:
    try:
        prompt = get_quote_highlight_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Extract short quote; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        return parsed.get("quote_highlight") or ""
    except Exception as e:
        logging.warning(f"Quote highlight failed: {e}")
        return ""


def assess_toxicity(client: AzureOpenAI, deployment: str, text: str) -> float:
    try:
        prompt = get_toxicity_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Assess toxicity; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        val = parsed.get("toxicity_score")
        try:
            return max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            return 0.0
    except Exception as e:
        logging.warning(f"Toxicity scoring failed: {e}")
        return 0.0


def get_tone_quality_flags(client: AzureOpenAI, deployment: str, text: str) -> Tuple[bool, bool, bool, float]:
    try:
        prompt = get_tone_quality_flags_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Tone/quality flags; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        sarcasm = _to_bool(parsed.get("sarcasm_flag"))
        humor = _to_bool(parsed.get("humor_flag"))
        spam = _to_bool(parsed.get("spam_flag"))
        coh = parsed.get("coherence_score")
        try:
            coh_val = max(0.0, min(1.0, float(coh)))
        except (TypeError, ValueError):
            coh_val = 0.5
        return sarcasm, humor, spam, coh_val
    except Exception as e:
        logging.warning(f"Tone/quality flags failed: {e}")
        return False, False, False, 0.5


def get_bug_info(client: AzureOpenAI, deployment: str, text: str) -> Tuple[bool, str, str]:
    try:
        prompt = get_bug_info_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Bug info; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        report = _to_bool(parsed.get("bug_report"))
        bug_type = (parsed.get("bug_type") or "none").strip().lower()
        allowed = {"crash","freeze","input","save","ui","audio","perf","network","none"}
        if bug_type not in allowed:
            bug_type = "none"
        steps = parsed.get("steps_hint") or ""
        return report, bug_type, steps
    except Exception as e:
        logging.warning(f"Bug info extraction failed: {e}")
        return False, "none", ""


def get_feature_request_bonus(client: AzureOpenAI, deployment: str, text: str) -> Tuple[bool, List[str], str]:
    try:
        prompt = get_feature_request_bonus_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Feature request bonus; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        fr = _to_bool(parsed.get("feature_request"))
        reqs = parsed.get("requested_features") or []
        reqs = [str(x).strip() for x in reqs if str(x).strip()] if isinstance(reqs, list) else []
        suggestion = parsed.get("suggestion_text") or ""
        return fr, reqs, suggestion
    except Exception as e:
        logging.warning(f"Feature request bonus failed: {e}")
        return False, [], ""


def get_nps_and_emotion(client: AzureOpenAI, deployment: str, text: str) -> Tuple[str, str]:
    try:
        prompt = get_nps_and_emotion_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "NPS & emotion; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        nps = (parsed.get("nps_category") or "").strip().lower()
        if nps not in {"detractor","passive","promoter"}:
            nps = "passive"
        emotion = (parsed.get("emotion_primary") or "").strip().lower()
        return nps, emotion
    except Exception as e:
        logging.warning(f"NPS/emotion extraction failed: {e}")
        return "passive", ""


def infer_playtime_bucket(client: AzureOpenAI, deployment: str, text: str) -> str:
    try:
        prompt = get_playtime_bucket_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Playtime bucket; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        bucket = (parsed.get("playtime_bucket") or "none").strip().lower()
        allowed = {"none","<30min","30–60min","1–5h","5–20h","20h+"}
        if bucket not in allowed:
            bucket = "none"
        return bucket
    except Exception as e:
        logging.warning(f"Playtime bucket inference failed: {e}")
        return "none"


def infer_reviewer_experience_level(client: AzureOpenAI, deployment: str, text: str) -> str:
    try:
        prompt = get_reviewer_experience_prompt(text or "")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Reviewer experience level; return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        lvl = (parsed.get("reviewer_experience_level") or "régulier").strip().lower()
        allowed = {"novice","régulier","core","hardcore"}
        if lvl not in allowed:
            lvl = "régulier"
        return lvl
    except Exception as e:
        logging.warning(f"Reviewer experience extraction failed: {e}")
        return "régulier"


def fetch_pending_reviews(conn: duckdb.DuckDBPyConnection, limit: int) -> List[Tuple[str, str]]:
    rows = conn.execute("""
        SELECT r.recommendationid, r.review
        FROM raw_reviews r
        LEFT JOIN llm_enrichment e ON e.recommendationid = r.recommendationid
        WHERE r.review IS NOT NULL
          AND (
            e.recommendationid IS NULL OR
            e.sentiment_label IS NULL OR
            e.sentiment_score IS NULL OR
            e.summary_10_words IS NULL OR
            e.tl_dr IS NULL OR
            e.keywords IS NULL OR
            e.themes IS NULL OR
            e.pros IS NULL OR
            e.cons IS NULL OR
            e.aspect_scores IS NULL OR
            e.feature_requests IS NULL OR
            e.language_detected IS NULL OR
            e.normalized_text_en IS NULL OR
            e.quote_highlight IS NULL OR
            e.toxicity_score IS NULL OR
            e.sarcasm_flag IS NULL OR
            e.humor_flag IS NULL OR
            e.spam_flag IS NULL OR
            e.coherence_score IS NULL OR
            e.bug_report IS NULL OR
            e.bug_type IS NULL OR
            e.steps_hint IS NULL OR
            e.feature_request IS NULL OR
            e.requested_features IS NULL OR
            e.suggestion_text IS NULL OR
            e.playtime_bucket IS NULL OR
            e.reviewer_experience_level IS NULL OR
            e.nps_category IS NULL OR
            e.emotion_primary IS NULL OR
            e.pertinence IS NULL
          )
        ORDER BY r.timestamp_created DESC
        LIMIT ?
    """, [limit]).fetchall()
    return rows




def insert_enrichment(
    conn: duckdb.DuckDBPyConnection,
    rec_id: str,
    label: str,
    score: float,
    summary_10_words: str,
    tl_dr: str,
    keywords_json: str,
    themes_json: str,
    pros_json: str,
    cons_json: str,
    aspect_scores_json: str,
    feature_requests_json: str,
    language_detected: str,
    normalized_text_en: str,
    # bonus set
    quote_highlight: str,
    toxicity_score: float,
    sarcasm_flag: bool,
    humor_flag: bool,
    spam_flag: bool,
    coherence_score: float,
    bug_report: bool,
    bug_type: str,
    steps_hint: str,
    feature_request: bool,
    requested_features_json: str,
    suggestion_text: str,
    playtime_bucket: str,
    reviewer_experience_level: str,
        nps_category: str,
    emotion_primary: str,
    pertinence: bool,
    force: bool = False,
) -> str:
    row = conn.execute(
        "SELECT sentiment_label, sentiment_score, summary_10_words, tl_dr, keywords, themes, pros, cons, aspect_scores, feature_requests, language_detected, normalized_text_en, quote_highlight, toxicity_score, sarcasm_flag, humor_flag, spam_flag, coherence_score, bug_report, bug_type, steps_hint, feature_request, requested_features, suggestion_text, playtime_bucket, reviewer_experience_level, nps_category, emotion_primary, pertinence FROM llm_enrichment WHERE recommendationid = ?",
        [rec_id]
    ).fetchone()

    if row is None:
        columns = [
            "recommendationid",
            "sentiment_label",
            "sentiment_score",
            "summary_10_words",
            "tl_dr",
            "keywords",
            "themes",
            "pros",
            "cons",
            "aspect_scores",
            "feature_requests",
            "language_detected",
            "normalized_text_en",
            "quote_highlight",
            "toxicity_score",
            "sarcasm_flag",
            "humor_flag",
            "spam_flag",
            "coherence_score",
            "bug_report",
            "bug_type",
            "steps_hint",
            "feature_request",
            "requested_features",
            "suggestion_text",
            "playtime_bucket",
            "reviewer_experience_level",
            "nps_category",
            "emotion_primary",
            "pertinence",
        ]
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO llm_enrichment ({', '.join(columns)}) VALUES ({placeholders})"
        params = [
            rec_id,
            label,
            score,
            summary_10_words,
            tl_dr,
            keywords_json,
            themes_json,
            pros_json,
            cons_json,
            aspect_scores_json,
            feature_requests_json,
            language_detected,
            normalized_text_en,
            quote_highlight,
            toxicity_score,
            sarcasm_flag,
            humor_flag,
            spam_flag,
            coherence_score,
            bug_report,
            bug_type,
            steps_hint,
            feature_request,
            requested_features_json,
            suggestion_text,
            playtime_bucket,
            reviewer_experience_level,
            nps_category,
            emotion_primary,
            pertinence,
        ]
        conn.execute(sql, params)
        return "inserted"
    else:
        cols = [
            ("sentiment_label", row[0], label),
            ("sentiment_score", row[1], score),
            ("summary_10_words", row[2], summary_10_words),
            ("tl_dr", row[3], tl_dr),
            ("keywords", row[4], keywords_json),
            ("themes", row[5], themes_json),
            ("pros", row[6], pros_json),
            ("cons", row[7], cons_json),
            ("aspect_scores", row[8], aspect_scores_json),
            ("feature_requests", row[9], feature_requests_json),
            ("language_detected", row[10], language_detected),
            ("normalized_text_en", row[11], normalized_text_en),
            ("quote_highlight", row[12], quote_highlight),
            ("toxicity_score", row[13], toxicity_score),
            ("sarcasm_flag", row[14], sarcasm_flag),
            ("humor_flag", row[15], humor_flag),
            ("spam_flag", row[16], spam_flag),
            ("coherence_score", row[17], coherence_score),
            ("bug_report", row[18], bug_report),
            ("bug_type", row[19], bug_type),
            ("steps_hint", row[20], steps_hint),
            ("feature_request", row[21], feature_request),
            ("requested_features", row[22], requested_features_json),
            ("suggestion_text", row[23], suggestion_text),
            ("playtime_bucket", row[24], playtime_bucket),
            ("reviewer_experience_level", row[25], reviewer_experience_level),
            ("nps_category", row[26], nps_category),
            ("emotion_primary", row[27], emotion_primary),
            ("pertinence", row[28], pertinence),
        ]
        if force:
            upd_cols = [
                "sentiment_label",
                "sentiment_score",
                "summary_10_words",
                "tl_dr",
                "keywords",
                "themes",
                "pros",
                "cons",
                "aspect_scores",
                "feature_requests",
                "language_detected",
                "normalized_text_en",
                "quote_highlight",
                "toxicity_score",
                "sarcasm_flag",
                "humor_flag",
                "spam_flag",
                "coherence_score",
                "bug_report",
                "bug_type",
                "steps_hint",
                "feature_request",
                "requested_features",
                "suggestion_text",
                "playtime_bucket",
                "reviewer_experience_level",
                "nps_category",
                "emotion_primary",
                "pertinence",
            ]
            set_clause = ", ".join([f"{c} = ?" for c in upd_cols])
            sql = f"UPDATE llm_enrichment SET {set_clause} WHERE recommendationid = ?"
            params = [
                label,
                score,
                summary_10_words,
                tl_dr,
                keywords_json,
                themes_json,
                pros_json,
                cons_json,
                aspect_scores_json,
                feature_requests_json,
                language_detected,
                normalized_text_en,
                quote_highlight,
                toxicity_score,
                sarcasm_flag,
                humor_flag,
                spam_flag,
                coherence_score,
                bug_report,
                bug_type,
                steps_hint,
                feature_request,
                requested_features_json,
                suggestion_text,
                playtime_bucket,
                reviewer_experience_level,
                nps_category,
                emotion_primary,
                pertinence,
                rec_id,
            ]
            conn.execute(sql, params)
            return "overwritten"
        else:
            updated_any = False
            for col_name, existing_val, new_val in cols:
                if existing_val is None or (isinstance(existing_val, str) and existing_val == ""):
                    conn.execute(
                        f"UPDATE llm_enrichment SET {col_name} = ? WHERE recommendationid = ?",
                        [new_val, rec_id]
                    )
                    updated_any = True
            return "updated" if updated_any else "skipped"







def main():
    parser = argparse.ArgumentParser(description="Enrich raw_reviews with core LLM indicators using Azure OpenAI.")
    parser.add_argument("--db-name", default=os.getenv("MD_DB_NAME", "steam_analytics"), help="MotherDuck DB name")
    parser.add_argument("--limit", type=int, default=50, help="Max number of reviews to classify this run")
    parser.add_argument("--force", action="store_true", help="Overwrite existing sentiment_label if already set")

    
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    conn = connect_motherduck(args.db_name)
    conn.execute(SCHEMA_SQL)

            # Ensure all core-set and bonus-set columns exist for older tables
    try:
        # core
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS sentiment_score DOUBLE")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS summary_10_words TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS tl_dr TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS keywords TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS themes TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS pros TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS cons TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS aspect_scores TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS feature_requests TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS language_detected TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS normalized_text_en TEXT")
        # bonus
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS quote_highlight TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS toxicity_score DOUBLE")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS sarcasm_flag BOOLEAN")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS humor_flag BOOLEAN")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS spam_flag BOOLEAN")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS coherence_score DOUBLE")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS bug_report BOOLEAN")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS bug_type TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS steps_hint TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS feature_request BOOLEAN")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS requested_features TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS suggestion_text TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS playtime_bucket TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS reviewer_experience_level TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS nps_category TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS emotion_primary TEXT")
        conn.execute("ALTER TABLE llm_enrichment ADD COLUMN IF NOT EXISTS pertinence BOOLEAN")
    except Exception as e:
        logging.debug(f"Column migration step encountered an issue: {e}")



    client = get_azure_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT not set")

    pending = fetch_pending_reviews(conn, args.limit)
    logging.info(f"Found {len(pending)} reviews pending sentiment classification.")

    processed = 0
    inserted = updated = overwritten = skipped = 0
    for rec_id, text in pending:
        # Detect language and normalize to English
        lang, norm_text = detect_language_and_normalize(client, deployment, text or "")
        norm_text = norm_text or (text or "")

        # Classify sentiment on normalized text
        label, score = classify_sentiment(client, deployment, norm_text)

        # Generate other core-set enrichments
        summary_10_words, tl_dr = generate_summary(client, deployment, norm_text)
        keywords, themes = extract_keywords_themes(client, deployment, norm_text)
        pros, cons = extract_pros_cons(client, deployment, norm_text)
        aspect_scores = score_aspects(client, deployment, norm_text)
        feature_requests = find_feature_requests(client, deployment, norm_text)

        # Bonus-set enrichments
        quote_highlight = generate_quote_highlight(client, deployment, norm_text)
        toxicity_score = assess_toxicity(client, deployment, norm_text)
        sarcasm_flag, humor_flag, spam_flag, coherence_score = get_tone_quality_flags(client, deployment, norm_text)
        bug_report, bug_type, steps_hint = get_bug_info(client, deployment, norm_text)
        feature_request, requested_features, suggestion_text = get_feature_request_bonus(client, deployment, norm_text)
        nps_category, emotion_primary = get_nps_and_emotion(client, deployment, norm_text)
        playtime_bucket = infer_playtime_bucket(client, deployment, norm_text)
        reviewer_experience_level = infer_reviewer_experience_level(client, deployment, norm_text)

        # Pertinence classification
        try:
            pr_prompt = get_pertinence_prompt(norm_text)
            pr_resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "Classify pertinence; return ONLY JSON."},
                    {"role": "user", "content": pr_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            pr_parsed = json.loads(pr_resp.choices[0].message.content or "{}")
            pertinence = _to_bool(pr_parsed.get("pertinence"))
        except Exception as e:
            logging.warning(f"Pertinence classification failed: {e}")
            pertinence = True

        # Serialize lists/maps to JSON strings for storage
        keywords_json = json.dumps(keywords, ensure_ascii=False)
        themes_json = json.dumps(themes, ensure_ascii=False)
        pros_json = json.dumps(pros, ensure_ascii=False)
        cons_json = json.dumps(cons, ensure_ascii=False)
        aspect_scores_json = json.dumps(aspect_scores, ensure_ascii=False)
        feature_requests_json = json.dumps(feature_requests, ensure_ascii=False)
        requested_features_json = json.dumps(requested_features, ensure_ascii=False)
        status = insert_enrichment(
            conn, rec_id, label, score,
            summary_10_words, tl_dr,
            keywords_json, themes_json,
            pros_json, cons_json,
            aspect_scores_json,
            feature_requests_json,
            lang, norm_text,
            # bonus
            quote_highlight,
            toxicity_score,
            sarcasm_flag,
            humor_flag,
            spam_flag,
            coherence_score,
            bug_report,
            bug_type,
            steps_hint,
            feature_request,
            requested_features_json,
            suggestion_text,
            playtime_bucket,
            reviewer_experience_level,
                        nps_category,
            emotion_primary,
            pertinence,
            force=args.force
        )
        if status == "inserted":
            inserted += 1
        elif status == "updated":
            updated += 1
        elif status == "overwritten":
            overwritten += 1
        else:
            skipped += 1
        processed += 1
        if processed % 10 == 0:
            logging.info(f"Processed {processed}/{len(pending)} (ins:{inserted}, upd:{updated}, ovw:{overwritten}, skip:{skipped})")

    print(f"Done. Classified {processed} reviews. Results: inserted={inserted}, updated={updated}, overwritten={overwritten}, skipped={skipped}. Total rows: {conn.execute('SELECT COUNT(*) FROM llm_enrichment').fetchone()[0]}")


if __name__ == "__main__":
    main()