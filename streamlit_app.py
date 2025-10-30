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


# Helpers
def _safe_parse_list_cell(cell):
    try:
        if pd.isna(cell):
            return []
        if isinstance(cell, list):
            return [str(x) for x in cell]
        # list stored as string → try to eval safely
        import ast
        parsed = ast.literal_eval(str(cell))
        return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else []
    except Exception:
        return []

def _explode_counts(df: pd.DataFrame, col: str, top_n: int = 12) -> pd.DataFrame:
    items = []
    if col in df.columns:
        for vals in df[col].apply(_safe_parse_list_cell).tolist():
            items.extend(vals)
    counts = pd.Series(items, dtype="object").value_counts().reset_index()
    counts.columns = [col, "count"]
    return counts.head(top_n)

def _parse_llm_score_mean(df: pd.DataFrame) -> pd.DataFrame:
    # Parse JSON-like score dict per row, compute mean per aspect
    import json
    import ast
    aspects_accum = {}
    counts = {}
    if "llm_score" not in df.columns:
        return pd.DataFrame({"aspect": [], "score": []})
    for cell in df["llm_score"].fillna("{}"):
        try:
            if isinstance(cell, dict):
                d = cell
            else:
                s = str(cell)
                try:
                    d = json.loads(s)
                except Exception:
                    # fallback for python-literal-like strings
                    d = ast.literal_eval(s)
            for k, v in d.items():
                try:
                    v_float = float(v)
                except Exception:
                    continue
                aspects_accum[k] = aspects_accum.get(k, 0.0) + v_float
                counts[k] = counts.get(k, 0) + 1
        except Exception:
            continue
    rows = []
    for k in aspects_accum:
        if counts.get(k, 0) > 0:
            rows.append({"aspect": str(k), "score": aspects_accum[k] / counts[k]})
    return pd.DataFrame(rows).sort_values("aspect")

# Carte "Nombre de reviews" avec courbe de tendance en fond dans la première colonne
cols_top = st.columns(6)
# Sidebar : filters
with st.sidebar:
    st.header("Chart parameters ⚙️")
    # Filtres rapides existants
    bug_reported = st.checkbox("Avis avec bugs signalés uniquement")
    recommend_only = st.checkbox("Avis positifs uniquement")
    purchased_only = st.checkbox("Avis avec achat uniquement")
    reviewer_catalog = st.checkbox("Reviewer posséde d'autres jeux")

    st.divider()
    st.subheader("Filtres avancés")

    # Plage de dates (création)
    min_date = pd.to_datetime(data["date_created"]).min().date() if len(data) else None
    max_date = pd.to_datetime(data["date_created"]).max().date() if len(data) else None
    date_range = st.date_input(
        "Période (date de création)",
        value=(min_date, max_date) if min_date and max_date else None,
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD"
    )

    # Langues
    languages = sorted([str(x) for x in data["language"].dropna().unique().tolist()])
    selected_languages = st.multiselect(
        "Langues",
        options=languages,
        default=languages
    )

    # Scores
    sentiment_min, sentiment_max = st.slider(
        "Score de sentiment (LLM)", 0.0, 1.0, (0.0, 1.0), 0.01
    )
    tox_min, tox_max = st.slider(
        "Score de toxicité (LLM)", 0.0, 1.0, (0.0, 1.0), 0.01
    )

    # Drapeaux LLM
    only_feature_req = st.checkbox("Inclure uniquement les demandes de fonctionnalités (LLM)")
    only_pertinent = st.checkbox("Avis jugés pertinents (LLM)")
    exclude_spam = st.checkbox("Exclure les avis flaggés spam (LLM)")

    # Recherche texte
    search_query = st.text_input("Recherche texte (original + traduit)")

    # Espace export
    st.divider()
    st.caption("Exports du jeu de données filtré")


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

# Appliquer filtres avancés
try:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
        start_date, end_date = date_range
        filtered_data = filtered_data[(filtered_data["date_created"] >= start_date) & (filtered_data["date_created"] <= end_date)]
except Exception:
    pass

if selected_languages:
    filtered_data = filtered_data[filtered_data["language"].astype(str).isin(selected_languages)]

filtered_data = filtered_data[
    (filtered_data["llm_sentiment_score"].fillna(0.0) >= sentiment_min)
    & (filtered_data["llm_sentiment_score"].fillna(0.0) <= sentiment_max)
]

filtered_data = filtered_data[
    (filtered_data["llm_toxicity_score"].fillna(0.0) >= tox_min)
    & (filtered_data["llm_toxicity_score"].fillna(0.0) <= tox_max)
]

if only_feature_req:
    filtered_data = filtered_data[filtered_data["llm_feature_requested_flag"] == True]
if only_pertinent:
    filtered_data = filtered_data[filtered_data["llm_review_pertinence_flag"] == True]
if exclude_spam:
    filtered_data = filtered_data[filtered_data["llm_spam_flag"] != True]

if search_query:
    q = str(search_query).strip()
    if q:
        mask = (
            filtered_data["review"].astype(str).str.contains(q, case=False, na=False)
            | filtered_data["llm_review_translated"].astype(str).str.contains(q, case=False, na=False)
        )
        filtered_data = filtered_data[mask]

# Afficher le nombre de lignes et proposer l'export (dans la sidebar)
with st.sidebar:
    st.metric("Lignes filtrées", value=f"{len(filtered_data):,}")
    csv_bytes = filtered_data.to_csv(index=False).encode("utf-8", "ignore")
    st.download_button("Télécharger CSV", data=csv_bytes, file_name="steam_reviews_filtered.csv", mime="text/csv")
    jsonl_str = filtered_data.to_json(orient="records", lines=True, force_ascii=False)
    st.download_button("Télécharger JSONL", data=jsonl_str, file_name="steam_reviews_filtered.jsonl", mime="application/json")


with cols_top[0]:
    st.subheader("Reviews")
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
    st.subheader("% reco")
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





# Section: Thématiques, émotions, aspects
st.subheader("Thématiques et qualité perçue")
st.caption("Barres: issues de la colonne `llm_themes`, `llm_pros`, `llm_cons` (LLM).")
g1, g2, g3 = st.columns([2, 2, 2])

with g1:
    st.caption("Top thèmes (LLM)")
    themes_counts = _explode_counts(filtered_data, "llm_themes", top_n=12)
    if not themes_counts.empty:
        row_count = len(themes_counts)
        chart_height = max(280, 28 * row_count)
        chart_themes = alt.Chart(themes_counts).mark_bar(size=22).encode(
            y=alt.Y("llm_themes:N", sort='-x', title=None, axis=alt.Axis(labelLimit=200, labelAngle=0, labelFontSize=13, labelAlign='right'), scale=alt.Scale(padding=5)),
            x=alt.X("count:Q", title=None, axis=None),
            color=alt.value("#A6D8A8")
        )
        labels = alt.Chart(themes_counts).mark_text(
            align="right", baseline="middle", dx=-6, color="white", fontSize=12
        ).encode(
            y=alt.Y("llm_themes:N", sort='-x'),
            x=alt.X("count:Q"),
            text=alt.Text("count:Q", format=",d")
        )
        st.altair_chart((chart_themes + labels).properties(height=chart_height), use_container_width=True)
    else:
        st.info("Aucun thème détecté dans la sélection.")

with g2:
    st.caption("Top Pros (LLM)")
    pros_counts = _explode_counts(filtered_data, "llm_pros", top_n=12)
    if not pros_counts.empty:
        row_count = len(pros_counts)
        chart_height = max(280, 28 * row_count)
        chart_pros = alt.Chart(pros_counts).mark_bar(size=22).encode(
            y=alt.Y("llm_pros:N", sort='-x', title=None, axis=alt.Axis(labelLimit=200, labelAngle=0, labelFontSize=13, labelAlign='right'), scale=alt.Scale(padding=5)),
            x=alt.X("count:Q", title=None, axis=None),
            color=alt.value("#B4D5F5")
        )
        labels = alt.Chart(pros_counts).mark_text(
            align="right", baseline="middle", dx=-6, color="white", fontSize=12
        ).encode(
            y=alt.Y("llm_pros:N", sort='-x'),
            x=alt.X("count:Q"),
            text=alt.Text("count:Q", format=",d")
        )
        st.altair_chart((chart_pros + labels).properties(height=chart_height), use_container_width=True)
    else:
        st.info("Aucun pro détecté dans la sélection.")

with g3:
    st.caption("Top Cons (LLM)")
    cons_counts = _explode_counts(filtered_data, "llm_cons", top_n=12)
    if not cons_counts.empty:
        row_count = len(cons_counts)
        chart_height = max(280, 28 * row_count)
        chart_cons = alt.Chart(cons_counts).mark_bar(size=22).encode(
            y=alt.Y("llm_cons:N", sort='-x', title=None, axis=alt.Axis(labelLimit=200, labelAngle=0, labelFontSize=13, labelAlign='right'), scale=alt.Scale(padding=5)),
            x=alt.X("count:Q", title=None, axis=None),
            color=alt.value("#F7B3B3")
        )
        labels = alt.Chart(cons_counts).mark_text(
            align="right", baseline="middle", dx=-6, color="white", fontSize=12
        ).encode(
            y=alt.Y("llm_cons:N", sort='-x'),
            x=alt.X("count:Q"),
            text=alt.Text("count:Q", format=",d")
        )
        st.altair_chart((chart_cons + labels).properties(height=chart_height), use_container_width=True)
    else:
        st.info("Aucun inconvénient détecté dans la sélection.")


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
        time_grain = st.radio(
            "Granularité (affichage)",
            options=["Jour","Mois"],
            index=0,
            horizontal=True,
            key="radio_granularite"
        )
        time_unit = "yearmonthdate" if time_grain == "Jour" else "yearmonth"

                # Barres empilées (compte) + ligne de moyenne cumulative recommandée (%) avec légende
        # Couleurs pastel pour recommandé (positif) et non recommandé (négatif)
        # Construit 'series' explicitement pour contrôler l'ordre d'empilement (non recommandé en haut, recommandé en bas)
        bars = alt.Chart(filtered_data).transform_calculate(
            series="datum.recommend_the_game ? 'Recommandé' : 'Non recommandé'"
        ).mark_bar().encode(
            x=alt.X(f"{time_unit}(date_created):T", title=None),
            y=alt.Y("count():Q", title=None),
            color=alt.Color(
                "series:N",
                title="Série",
                sort=["Non recommandé", "Recommandé"],
                scale=alt.Scale(
                    domain=["Non recommandé", "Recommandé", "Moyenne cumulative (%)"],
                    range=["#F7B3B3", "#A6D8A8", "#1f77b4"]
                )
            ),
            order=alt.Order(
                "series:N",
                sort="ascending"
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
                y=alt.Y("percent_cum:Q", title=None, axis=alt.Axis(title=None)),
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
        # Palette pastel étendue pour couvrir plus d'émotions, et fallback cyclique
        emotion_palette = {
            "joy": "#A6D8A8",
            "sadness": "#B4D5F5",
            "anger": "#F7B3B3",
            "surprise": "#FAEDA3",
            "fear": "#B8A4E3",
            "disgust": "#FFD1B3",
            "trust": "#C7EDCC",
            "anticipation": "#F6DDCC",
            "contempt": "#CFC7CF",
            "boredom": "#B7D7E8",
            "excitement": "#E7CEF7",
            "admiration": "#DDEBCF",
            "neutral": "#CCCCCC",
            "unknown": "#CCCCCC",
            "undefined": "#CCCCCC",
        }
        pastel_cycle_extra = ["#FCD5CE","#B8F2E6","#CABBE9","#FFF1E6","#B6E2D3","#E1C3E6"]
        emo_df = filtered_data["llm_emotion"].fillna("unknown").astype(str).value_counts().reset_index()
        emo_df.columns = ["llm_emotion", "count"]
        emo_df = emo_df.sort_values("count", ascending=False)
        emo_labels = emo_df["llm_emotion"].tolist()
        # Color mapping avec fallback cyclique pastel
        color_domain = emo_labels
        # Associer chaque label à une couleur unique
        used_for_others = 0
        color_range = []
        for l in emo_labels:
            base = emotion_palette.get(l.lower())
            if base is not None:
                color_range.append(base)
            else:
                # Pastel unique et cyclique pour les hors liste
                color_range.append(pastel_cycle_extra[used_for_others % len(pastel_cycle_extra)])
                used_for_others += 1
        bars = alt.Chart(emo_df).mark_bar().encode(
            y=alt.Y("llm_emotion:N", title=None, sort=emo_labels, axis=alt.Axis(labelLimit=150, labelAngle=0, labelFontSize=13, labelAlign="right")),
            x=alt.X("count:Q", title=None, axis=None),
            color=alt.Color("llm_emotion:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None)
        )
        labels = alt.Chart(emo_df).mark_text(
            align="right", baseline="middle", dx=-6, color="white", fontSize=12
        ).encode(
            y=alt.Y("llm_emotion:N", sort=emo_labels),
            x=alt.X("count:Q"),
            text=alt.Text("count:Q", format=",d")
        )
        st.altair_chart((bars + labels).properties(height=max(280, 28*len(emo_df))), use_container_width=True)


    st.subheader("Tableau des avis")
        # Affichage du tableau (avec word-wrap sur le texte traduit)
        # Préparer affichage avec wrap multi-lignes sur la colonne 'Avis traduit'
    display_df = filtered_data.copy()
    display_df["llm_review_translated"] = display_df["llm_review_translated"].astype(str).apply(lambda s: fill(s, width=120))

    sorted_df = display_df.sort_values(by="date_updated", ascending=False).reset_index(drop=True)
    sorted_df["rid_str"] = sorted_df["recommendation_ID"].astype(str)
    # Colonne de sélection (single choice) et position en première colonne
    selected_rid_state = st.session_state.get("selected_rid")
    sorted_df["select"] = sorted_df["recommendation_ID"].astype(str).eq(str(selected_rid_state))
    # Réordonner pour avoir 'select' en premier
    select_first_cols = ["select"] + [c for c in sorted_df.columns if c != "select"]
    sorted_df = sorted_df[select_first_cols]

    edited_df = st.data_editor(
        sorted_df,
        use_container_width=True,
        height=500,
        hide_index=True,
        disabled=False,
        column_config={
            "select": st.column_config.CheckboxColumn(
                "Sélection",
                help="Cliquer pour sélectionner une ligne",
                width=80
            ),
            "llm_review_translated": st.column_config.TextColumn(
                "Avis traduit",
                help="Avis traduit en anglais si langue <> en, fr",
                width=900,
                disabled=True
            )
        }
    )

    # Mettre à jour la sélection depuis la table (une seule ligne)
    selected_ids = edited_df.loc[edited_df["select"] == True, "rid_str"].astype(str).tolist()
    if selected_ids:
        # Forcer single-choice: garder la dernière sélection utilisateur
        st.session_state["selected_rid"] = selected_ids[-1]

    # Panneau de détail
    st.markdown("---")
    st.subheader("Détails de l'avis")
    row = None
    rid = st.session_state.get("selected_rid")
    if rid is not None and len(sorted_df):
        match = sorted_df[sorted_df["rid_str"].astype(str) == str(rid)]
        if not match.empty:
            row = match.iloc[0]
    if row is None and len(sorted_df):
        row = sorted_df.iloc[0]
        st.session_state["selected_rid"] = row["rid_str"]

    if row is not None:
        view_mode = st.radio("Texte à afficher", ["Avis traduit", "Original"], index=0, horizontal=True)
        text = row["llm_review_translated"] if view_mode == "Avis traduit" else row["review"]
        st.write(text)

        # Chips/Badges via markdown simple
        def _fmt_badge(label, color):
            return f"<span style='background:{color};padding:2px 6px;border-radius:8px;margin-right:6px;color:white;font-size:12px'>{label}</span>"

        flags = []
        if row.get("llm_spam_flag", False):
            flags.append(_fmt_badge("spam", "#d62728"))
        if row.get("llm_sarcasm_flag", False):
            flags.append(_fmt_badge("sarcasme", "#9467bd"))
        if row.get("llm_humor_flag", False):
            flags.append(_fmt_badge("humour", "#2ca02c"))
        if row.get("llm_bug_reported_flag", False):
            flags.append(_fmt_badge("bug", "#e377c2"))
        if row.get("llm_feature_requested_flag", False):
            flags.append(_fmt_badge("feature", "#8c564b"))
        if flags:
            st.markdown(" ".join(flags), unsafe_allow_html=True)

        # Tags
        def _to_tags(cell):
            vals = _safe_parse_list_cell(cell)
            if not vals:
                return ""
            return ", ".join(sorted(set([str(v) for v in vals])))

        with st.expander("Mots-clés et thématiques", expanded=True):
            st.markdown(f"**Thèmes:** {_to_tags(row.get('llm_themes'))}")
            st.markdown(f"**Mots-clés:** {_to_tags(row.get('llm_keywords'))}")
            if pd.notna(row.get("llm_pros", None)) and str(row.get("llm_pros")) != "":
                st.markdown(f"**Pros:** {row.get('llm_pros')}")
            if pd.notna(row.get("llm_cons", None)) and str(row.get("llm_cons")) != "":
                st.markdown(f"**Cons:** {row.get('llm_cons')}")

        with st.expander("Métadonnées", expanded=True):
            meta_cols = st.columns(2)
            with meta_cols[0]:
                st.markdown(f"- Langue: `{row['language']}`")
                st.markdown(f"- Créé: `{row['date_created']}`")
                st.markdown(f"- Mis à jour: `{row['date_updated']}`")
                st.markdown(f"- Recommandé: `{bool(row['recommend_the_game'])}`")
                st.markdown(f"- Jeux possédés (auteur): `{row.get('author_num_games_owned', '')}`")
            with meta_cols[1]:
                st.markdown(f"- Votes +: `{row['count_review_liked']}`")
                st.markdown(f"- Funny: `{row['count_review_marked_funny']}`")
                st.markdown(f"- Toxicité: `{row.get('llm_toxicity_score', 0)}`")
                st.markdown(f"- NPS: `{row.get('llm_NPS', '')}`")
                st.markdown(f"- Avis (auteur): `{row.get('author_num_reviews', '')}`")
            st.markdown(f"- Temps de jeu total (auteur): `{row.get('author_playtime_forever', '')}`")



filtered_reviews_table(filtered_data)


st.subheader("Demandes de fonctionnalités — Pivot par tag")
# Préparer les données: tags, texte de demande, avis traduit
fr_df = filtered_data.copy()
fr_df["tags_list"] = fr_df["llm_feature_requested_tag"].apply(_safe_parse_list_cell)
fr_df["suggestion_text_str"] = fr_df["llm_feature_requested_text"].astype(str)
fr_df["review_translated_str"] = fr_df["llm_review_translated"].astype(str)
fr_df = fr_df[(fr_df["tags_list"].map(len) > 0) | (fr_df["suggestion_text_str"].str.strip() != "") | (fr_df["review_translated_str"].str.strip() != "")]

# Table plate (une ligne par tag/review)
records = []
for _, r in fr_df.iterrows():
    tags = r["tags_list"] if isinstance(r["tags_list"], list) and len(r["tags_list"]) else []
    suggestion = str(r.get("llm_feature_requested_text", "")).strip()
    review = str(r.get("llm_review_translated", "")).strip()
    if not tags:
        tags = ["(sans tag)"]
    for t in tags:
        records.append({"tag": str(t), "suggestion": suggestion, "review": review})

details_df = pd.DataFrame(records)

if not details_df.empty:
    # Joindre en listes uniques séparées par des sauts de ligne
    def _join_unique(values):
        seen = set()
        out = []
        for v in values:
            s = str(v).strip()
            if not s:
                continue
            if s not in seen:
                out.append(s)
                seen.add(s)
        return "\n".join(out)

    pivot_df = details_df.groupby("tag").agg(
        occurrences=("tag", "size"),
        feature_request_text=("suggestion", _join_unique),
        avis_llm_review_translated=("review", _join_unique)
    ).reset_index().sort_values("occurrences", ascending=False)

    st.dataframe(
        pivot_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "tag": st.column_config.TextColumn("Tag", width=260),
            "occurrences": st.column_config.NumberColumn("Nbr apparitions", width=160),
            "feature_request_text": st.column_config.TextColumn("Feature request text", width=600),
            "avis_llm_review_translated": st.column_config.TextColumn("Avis (llm_review_translated)", width=600)
        }
    )

    csv_bytes = pivot_df.to_csv(index=False).encode("utf-8", "ignore")
    st.download_button(
        label="Télécharger pivot CSV",
        data=csv_bytes,
        file_name="feature_requests_pivot.csv",
        mime="text/csv"
    )
else:
    st.info("Aucune demande de fonctionnalité détectée dans les reviews filtrées.")

