# Liste des indicateurs ajoutés via LLM

[Core set (prioritaire)]
sentiment_label: positive | neutral | negative
sentiment_score: 0..1 (confiance/force du sentiment global)
summary_10_words: résumé ultra court en ~10 mots
tl_dr: résumé 1 phrase (70–120 caractères)
keywords: liste de mots-clés principaux (3–8)
themes: liste de thèmes (3–5) au niveau macro (ex: gameplay, bugs, performance, difficulté, prix, histoire, contrôles, audio, art)
pros: liste de points positifs extraits
cons: liste de points négatifs extraits
aspect_scores: JSON map {aspect: score} avec score dans [-1, 1]
Exemples d’aspects: gameplay, controls, performance, bugs, visuals/art, audio, story, difficulty, content/length, price/value, tutorial/onboarding, UI/UX, multiplayer/netcode
feature_requests: liste des demandes de features (ex: “coop local”, “rebind des touches”, “mode facile”)
language_detected: langue détectée (fr, en, etc.)
normalized_text_en: version traduite du commentaire en français (si original en français) ou anglais si n'importe quelle autre langue

[bonus set]
quote_highlight: phrase courte “quote” représentative à afficher (ex: 12–20 mots)
toxicity_score: 0..1 (toxicité/insultes)
sarcasm_flag: bool
humor_flag: bool
spam_flag: bool (pub, répétitif, hors-sujet)
coherence_score: 0..1 (qualité/compréhensibilité du texte)
bug_report: bool (si la review contient un bug)
bug_type: catégorie du bug (crash, freeze, input, save, UI, audio, perf, network)
steps_hint: court texte d’indice de reproduction (“freeze au boss 2 quand on utilise dash en l’air”)
feature_request: bool
requested_features: liste (idem feature_requests ci-dessus)
suggestion_text: courte suggestion priorisée (1–2 phrases)
playtime_bucket: none | <30min | 30–60min | 1–5h | 5–20h | 20h+ (à partir de playtime_forever/playtime_at_review)
reviewer_experience_level: novice | régulier | core | hardcore (heuristique via num_games_owned, num_reviews, playtime)
nps_category: detractor | passive | promoter (interprétation LLM)
emotion_primary: ex: joy | anger | disappointment | excitement | frustration
