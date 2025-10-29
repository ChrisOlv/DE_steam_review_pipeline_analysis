import os
import duckdb
import streamlit as st
import altair as alt
# Configurer la page Streamlit
st.set_page_config(page_title="Steam Reviews – Visualisation des données", layout="wide")

# Vérifier le token MotherDuck
token = os.getenv("MOTHERDUCK_TOKEN")
if not token:
    st.error("MOTHERDUCK_TOKEN manquant. Définissez-le comme variable ou secret.")
    st.stop()
os.environ["MOTHERDUCK_TOKEN"] = token.strip()

# Connexion à MotherDuck
@st.cache_resource
def get_db_connection():
    con = duckdb.connect("md:")
    con.execute("USE steam_analytics")  # Nom de la base, modifiable si nécessaire
    return con

conn = get_db_connection()

# Métriques Moyennes
def average_metrics():
    metrics = conn.execute(
        """
        SELECT ROUND(AVG(sentiment_score), 2) AS avg_sentiment,
               ROUND(AVG(toxicity_score), 2) AS avg_toxicity,
               ROUND(AVG(coherence_score), 2) AS avg_coherence
        FROM llm_enrichment
        """
    ).df()

    st.subheader("Métriques Moyennes – Score global")
    st.caption("Ces scores représentent des moyennes calculées à partir des avis enrichis :")
    st.dataframe(metrics.style.format({
        "avg_sentiment": "{:.2f}",
        "avg_toxicity": "{:.2f}",
        "avg_coherence": "{:.2f}"
    }), use_container_width=True)

# Visualisation de la répartition des sentiments
def sentiment_distribution():
    data = conn.execute(
        """
        SELECT sentiment_label, COUNT(*) AS count
        FROM llm_enrichment
        WHERE sentiment_label IS NOT NULL
        GROUP BY sentiment_label
        ORDER BY count DESC
        """
    ).df()

    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("sentiment_label", title="Sentiment"),
        y=alt.Y("count", title="Nombre"),
        color=alt.Color("sentiment_label", legend=None)
    ).properties(title="Répartition des sentiments dans les avis")

    # Ajout des labels sur les barres
    text = chart.mark_text(dy=-10).encode(text="count")
    st.altair_chart(chart + text, use_container_width=True)

# Visualisation de l'évolution des avis positifs
def reviews_evolution():
    data = conn.execute(
        """
        SELECT DATE(timestamp_created) as review_date,
               AVG(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) * 100 AS positive_rate
        FROM llm_enrichment
        GROUP BY review_date
        ORDER BY review_date
        """
    ).df()

    chart = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("review_date:T", title="Date"),
        y=alt.Y("positive_rate:Q", title="Pourcentage de positifs (%)")
    ).properties(
        title="Évolution des évaluations positives dans les avis"
    )

    st.altair_chart(chart, use_container_width=True)

# Visualisation de la répartition des émotions détectées
def emotions_distribution():
    data = conn.execute(
        """
        SELECT emotion_primary, COUNT(*) AS count
        FROM llm_enrichment
        WHERE emotion_primary IS NOT NULL
        GROUP BY emotion_primary
        ORDER BY count DESC
        """
    ).df()

    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("emotion_primary:N", title="Émotion principale"),
        y=alt.Y("count:Q", title="Nombre d'instances"),
        color="emotion_primary"
    ).properties(
        title="Répartition des émotions détectées"
    )

    st.altair_chart(chart, use_container_width=True)

# Tableau interactif avec filtres
def filtered_reviews():
    st.subheader("Avis enrichis – Tableau interactif")

    # Filtrage dynamique
    sentiment_filter = st.multiselect(
        "Filtrer par sentiments", options=["positive", "neutral", "negative"], default=[]
    )
    bug_report_filter = st.checkbox("Afficher seulement les avis avec des bugs reportés", value=False)

    # Construction de la requête selon les filtres
    where_clause = "WHERE 1=1 "
    if sentiment_filter:
        sentiments = ", ".join([f"'{s}'" for s in sentiment_filter])
        where_clause += f"AND sentiment_label IN ({sentiments}) "
    if bug_report_filter:
        where_clause += "AND bug_report = TRUE "

    query = f"""
        SELECT timestamp_created, review, sentiment_label, emotion_primary, bug_report
        FROM llm_enrichment
        {where_clause}
        ORDER BY timestamp_created DESC
        LIMIT 50
    """

    data = conn.execute(query).df()
    st.dataframe(data, use_container_width=True)

st.title("Steam Reviews – Visualisations améliorées")

# Appeler les visualisations
average_metrics()
sentiment_distribution()
reviews_evolution()
emotions_distribution()
filtered_reviews()

