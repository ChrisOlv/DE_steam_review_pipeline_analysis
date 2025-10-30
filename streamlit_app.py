import os
import duckdb
import streamlit as st
import altair as alt
import pandas as pd
from dotenv import load_dotenv
load_dotenv() 

# Configurer la page Streamlit
st.set_page_config(page_title="Steam Reviews Analytics", layout="wide")

# Vérifier le token MotherDuck
token = os.getenv("MOTHERDUCK_TOKEN")
if not token:
    st.error("MOTHERDUCK_TOKEN manquant. Définissez-le comme variable ou secret.")
    st.stop()

# Connexion à MotherDuck
@st.cache_resource
def get_db_connection():
    con = duckdb.connect("md:")
    db_name = os.getenv("MD_DB_NAME", "steam_analytics")  # Nom de base de données par défaut
    con.execute(f"USE {db_name}")
    return con

conn = get_db_connection()

# Charger toute la table en cache
@st.cache_data(ttl=600)  # Cache pour 10 minutes (TTL: Time to Live)
def load_full_data():
    query = """
        SELECT 
               CAST(TO_TIMESTAMP(rr.timestamp_created) AS DATE) AS date_created,
               CAST(TO_TIMESTAMP(rr.timestamp_updated) AS DATE) AS date_updated,
               rr.recommendationid as recommendation_ID,
               rr.author_steamid as author_ID,
               rr.language,
               rr.review,
               rr.voted_up as recommend_the_game,
               rr.votes_up as count_review_liked,
               rr.votes_funny as count_review_marked_funny,
               rr.weighted_vote_score,
               rr.comment_count,
               rr.steam_purchase,
               rr.received_for_free,
               rr.author_num_games_owned,
               rr.author_num_reviews,
               rr.author_playtime_forever,
               rr.author_playtime_last_two_weeks,
               rr.author_last_played,
               le.sentiment_label as llm_sentiment_label,
               le.sentiment_score as llm_sentiment_score,
               le.summary_10_words as llm_10_words_summary,
               le.tl_dr,
               le.keywords as llm_keywords,
               le.themes as llm_themes,
               le.pros as llm_pros,
               le.cons as llm_cons,
               le.aspect_scores as llm_score,
               le.feature_requests as llm_feature_requests,
               le.language_detected,
               le.normalized_text_en as llm_review_translated,
               le.quote_highlight,
               le.toxicity_score as llm_toxicity_score,
               le.sarcasm_flag as llm_sarcasm_flag,
               le.humor_flag as llm_humor_flag,
               le.spam_flag as llm_spam_flag,
               le.coherence_score,
               le.bug_report as llm_bug_reported_flag,
               le.bug_type,
               le.steps_hint as llm_bug_report_text,
               le.feature_request as llm_feature_requested_flag,
               le.requested_features as llm_feature_requested_tag,
               le.suggestion_text as llm_feature_requested_text,
               le.playtime_bucket,
               le.reviewer_experience_level,
               le.nps_category as llm_NPS,
               le.emotion_primary as llm_emotion,
               le.pertinence as llm_review_pertinence_flag
        FROM raw_reviews rr
        JOIN llm_enrichment le ON rr.recommendationid = le.recommendationid
    """
    return conn.execute(query).df()

data = load_full_data()  # Charge toute la table

st.title("Steam Reviews Analytics – Visualisations basées sur OBT_data.sql")



# Exemple de tableau interactif avec filtres
def filtered_reviews_table(data):
    st.subheader("Tableau des avis enrichis avec filtres")
    
    # Ajouter des widgets pour le filtrage dynamique
    recommandation_options = st.multiselect(
        "Filtrer par recommandation", options=[True,False], default=[]
    )
    bug_reported = st.checkbox("Afficher seulement les avis avec des bugs signalés")

    # Filtrage des données
    filtered_data = data
    if recommandation_options:
        filtered_data = filtered_data[filtered_data["recommend_the_game"].isin(recommandation_options)]
    if bug_reported:
        filtered_data = filtered_data[filtered_data["llm_bug_reported_flag"] == True]

    st.dataframe(filtered_data, use_container_width=True)

filtered_reviews_table(data)