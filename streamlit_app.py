import os
import duckdb
import streamlit as st
import altair as alt
import pandas as pd
from textwrap import fill
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
               rr.language,
               le.normalized_text_en as llm_review_translated,
               rr.voted_up as recommend_the_game,
               rr.votes_up as count_review_liked,
               rr.votes_funny as count_review_marked_funny,
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
               le.pertinence as llm_review_pertinence_flag,
               rr.review,
               rr.recommendationid as recommendation_ID,
               ROUND(rr.weighted_vote_score,1) as weighted_vote_score,
               rr.author_steamid as author_ID
        FROM raw_reviews rr
        JOIN llm_enrichment le ON rr.recommendationid = le.recommendationid
    """
    
    df = conn.execute(query).df()
    # convertion des datatypes
    df["date_created"] = df["date_created"].dt.date  
    df["date_updated"] = df["date_updated"].dt.date 
    return df

data = load_full_data()  # Charge toute la table

st.title("🍕 Steam Reviews Analytics")
st.markdown(
    """
Ce tableau fournit une analyse approfondie des avis utilisateurs sur Steam, enrichis avec les capacités d'Azure OpenAI.

Les retours Steam sont extraits et mis à jour automatiquement chaque heure, stockés et analysés dans une base de données MotherDuck.

Le dépôt complet est disponible ici : [repository GitHub](https://github.com/ChrisOlv/DE_steam_review_pipeline_analysis).
    """
)


# Carte "Nombre de reviews" avec courbe de tendance en fond dans la première colonne
cols_top = st.columns(6)
# Sidebar : filters
with st.sidebar:
    st.header("Chart parameters ⚙️")
    bug_reported = st.checkbox("Avis avec bugs signalés uniquement")
    recommend_only = st.checkbox("Avis positifs uniquement")
    purchased_only = st.checkbox("Avis avec achat uniquement")
    reviewer_catalog = st.checkbox("Reviewer posséde d'autres jeux")


# Appliquer les filtres globalement
filtered_data = data
if bug_reported:
    filtered_data = filtered_data[filtered_data["llm_bug_reported_flag"] == True]
if recommend_only:
    filtered_data = filtered_data[filtered_data["recommend_the_game"] == True]
if purchased_only:
    filtered_data = filtered_data[filtered_data["received_for_free"] == False]
if reviewer_catalog:
    filtered_data = filtered_data[filtered_data["author_num_games_owned"] >= 1]

with cols_top[0]:
    st.subheader("Nombre de reviews")
    trend_df = (
        filtered_data.groupby("date_created").size().reset_index(name="count").sort_values("date_created")
    )

    if not trend_df.empty:
        max_count = int(trend_df["count"].max())
        label_df = pd.DataFrame({
            "label": [f"{len(filtered_data):,}"],
            "label_x": [trend_df["date_created"].min()],
            "label_y": [max_count * 0.95]
        })

        area = alt.Chart(trend_df).mark_area(opacity=0.25, color="#4c78a8").encode(
            x=alt.X("date_created:T", axis=None),
            y=alt.Y("count:Q", axis=None)
        )
        line = alt.Chart(trend_df).mark_line(color="#4c78a8", strokeWidth=2).encode(
            x="date_created:T",
            y="count:Q"
        )
        label = alt.Chart(label_df).mark_text(
            fontSize=28, fontWeight="bold", color="black", align="left"
        ).encode(
            x="label_x:T",
            y="label_y:Q",
            text="label:N"
        )

        card = (area + line + label).properties(height=120)
        st.altair_chart(card, use_container_width=True)
    else:
        st.metric("Nombre de reviews", value=f"{len(filtered_data):,}")

with cols_top[1]:
    st.subheader("% recommandé")
    rec_trend_df = (
        filtered_data.groupby("date_created").agg(
            total=("recommend_the_game", "size"),
            recommended=("recommend_the_game", "sum")
        ).reset_index().sort_values("date_created")
    )

    percent_total = ((filtered_data["recommend_the_game"] == True).sum() / len(filtered_data) * 100) if len(filtered_data) > 0 else 0

    if not rec_trend_df.empty:
        rec_trend_df["percent"] = rec_trend_df["recommended"] / rec_trend_df["total"] * 100
        max_percent = float(rec_trend_df["percent"].max())
        label_df = pd.DataFrame({
            "label": [f"{percent_total:.1f}%"],
            "label_x": [rec_trend_df["date_created"].min()],
            "label_y": [max_percent * 0.95 if max_percent > 0 else 0]
        })

        area = alt.Chart(rec_trend_df).mark_area(opacity=0.25, color="#72b67a").encode(
            x=alt.X("date_created:T", axis=None),
            y=alt.Y("percent:Q", axis=None, scale=alt.Scale(domain=[0, 100]))
        )
        line = alt.Chart(rec_trend_df).mark_line(color="#72b67a", strokeWidth=2).encode(
            x="date_created:T",
            y="percent:Q"
        )
        label = alt.Chart(label_df).mark_text(
            fontSize=28, fontWeight="bold", color="black", align="left"
        ).encode(
            x="label_x:T",
            y="label_y:Q",
            text="label:N"
        )

        card2 = (area + line + label).properties(height=120)
        st.altair_chart(card2, use_container_width=True)
    else:
        st.metric("Pourcentage recommandé", value=f"{percent_total:.1f}%")






# Tableau interactif avec filtres amélioré
def filtered_reviews_table(filtered_data):
    


# Cartes de métriques
    total_reviews = len(filtered_data)
    recommended_count = (filtered_data["recommend_the_game"] == True).sum()
    recommend_percent = (recommended_count / total_reviews *100 ) if total_reviews > 0 else 0

    # m1, m2 = st.columns(2)
    # with m1:
    #     st.metric("Nombre d'avis", value=f"{total_reviews:,}")
    # with m2:
    #     st.metric("Pourcentage recommandé", value=f"{recommend_percent:.1f}%")

            # Bar chart empilé: avis par date et recommandation + à droite: répartition par émotion
    st.subheader("Évolution des avis par date et recommandation")
    col_chart, col_emotion = st.columns([2, 1])

    with col_chart:
        st.subheader("Timeserie des avis")
        time_grain = st.selectbox("Granularité temporelle", options=["Mois","Jour"], index=1)
        time_unit = "yearmonth" if time_grain == "Mois" else "yearmonthdate"

                # Barres empilées (compte) + ligne de moyenne cumulative recommandée (%)
                # Barres empilées (compte) + ligne bleue de moyenne cumulative recommandée (%) avec légende
        # Couleurs pastel pour recommandé (positif) et non recommandé (négatif)
        bars = alt.Chart(filtered_data).transform_calculate(
            series="datum.recommend_the_game ? 'Recommandé' : 'Non recommandé'"
        ).mark_bar().encode(
            x=alt.X(f"{time_unit}(date_created):T", title=None),
            y=alt.Y("count():Q", title=None),
            color=alt.Color(
                "series:N",
                title="Série",
                scale=alt.Scale(
                    domain=["Recommandé", "Non recommandé", "Moyenne cumulative (%)"],
                    range=["#A6D8A8", "#F7B3B3", "#1f77b4"]
                )
            )
        )

        # Calcul de la moyenne cumulative jusqu'à chaque date (en %)
        period_series = pd.to_datetime(filtered_data["date_created"]) if time_grain == "Jour" else pd.to_datetime(filtered_data["date_created"]).dt.to_period("M").dt.to_timestamp()
        rec_df = (
            filtered_data.assign(period=period_series)
            .groupby("period")
            .agg(total=("recommend_the_game", "size"), recommended=("recommend_the_game", "sum"))
            .reset_index()
            .sort_values("period")
        )
        if not rec_df.empty:
            rec_df["cum_total"] = rec_df["total"].cumsum()
            rec_df["cum_recommended"] = rec_df["recommended"].cumsum()
            rec_df["percent_cum"] = rec_df["cum_recommended"] / rec_df["cum_total"] * 100
            rec_df["series"] = "Moyenne cumulative (%)"

            # Ligne bleue avec légende
            line = alt.Chart(rec_df).mark_line(strokeWidth=2).encode(
                x=alt.X("period:T", title=None),
                y=alt.Y("percent_cum:Q", title=None),
                color=alt.Color("series:N", title="Série")
            )

            # Points et étiquettes pour min, max et actuel
            min_idx = int(rec_df["percent_cum"].idxmin())
            max_idx = int(rec_df["percent_cum"].idxmax())
            current_idx = len(rec_df) - 1
            label_points = rec_df.loc[[min_idx, max_idx, current_idx]].copy()
            label_points["label"] = label_points["percent_cum"].map(lambda v: f"{v:.1f}%")

            points = alt.Chart(label_points).mark_point(color="#1f77b4").encode(
                x="period:T",
                y="percent_cum:Q"
            )
            labels = alt.Chart(label_points).mark_text(
                fontSize=11, dy=-8, color="#1f77b4"
            ).encode(
                x="period:T",
                y="percent_cum:Q",
                text="label:N"
            )

            chart = alt.layer(bars, line, points, labels).resolve_scale(y='independent').properties(height=350)
        else:
            chart = bars.properties(height=350)

        st.altair_chart(chart, use_container_width=True)



    with col_emotion:
        st.subheader("Répartition des avis par émotion")
        bars = alt.Chart(filtered_data).mark_bar().encode(
            y=alt.Y("llm_emotion:N", title=None, sort="-x"),
            x=alt.X("count():Q", title=None),
            color=alt.Color("llm_emotion:N", legend=None)
        )
        labels = alt.Chart(filtered_data).mark_text(
            align="left",
            dx=3,
            color="black"
        ).encode(
            y=alt.Y("llm_emotion:N", sort="-x"),
            x=alt.X("count():Q"),
            text=alt.Text("count():Q", format=",d")
        )
        emotion_chart = (bars + labels).properties(height=350)

        st.altair_chart(emotion_chart, use_container_width=True)


    st.subheader("Tableau des avis")
        # Affichage du tableau (avec word-wrap sur le texte traduit)
        # Préparer affichage avec wrap multi-lignes sur la colonne 'Avis traduit'
    display_df = filtered_data.copy()
    display_df["llm_review_translated"] = display_df["llm_review_translated"].astype(str).apply(lambda s: fill(s, width=120))

    st.data_editor(
        display_df.sort_values(by="date_updated", ascending=False),
        use_container_width=True,
        height=800,
        hide_index=True,
        disabled=True,
        column_config={
            "llm_review_translated": st.column_config.TextColumn(
                "Avis traduit",
                help="Avis traduit en anglais si langue <> en, fr",
                width=900
            )
        }
    )



filtered_reviews_table(filtered_data)

