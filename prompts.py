def get_sentiment_prompt(review_text):
    return (f"You are a strict sentiment classifier for Steam game reviews. "
            f"Return ONLY a JSON object with keys: "
            f"'sentiment_label' (one of 'positive', 'neutral', 'negative') and "
            f"'sentiment_score' (a number between 0 and 1 representing confidence/strength of sentiment). "
            f"Review:\n{review_text}")


def get_summary_prompt(review_text):
    return (f"Summarize the sentiment and main idea of the following Steam game review: "
            f"Output only as JSON format with keys: "
            f"'summary_10_words' (a summary not exceeding 10 words) "
            f"and 'tl_dr' (a one-sentence summary of 70–120 characters). "
            f"Review:\n{review_text}")


def get_keywords_and_themes_prompt(review_text):
    return (f"Analyze the following Steam review and identify key topics. Return ONLY a JSON object with keys: "
            f"'keywords': list 3-8 important keywords, and "
            f"'themes': list up to 5 broad themes. "
            f"Review:\n{review_text}")


def get_pros_and_cons_prompt(review_text):
    return (f"Review the following Steam game comment and extract clear pros and cons. "
            f"Return ONLY a JSON object with keys: "
            f"'pros': list of positive points extracted clearly, and "
            f"'cons': list of negative points extracted clearly. "
            f"Review:\n{review_text}")


def get_aspect_scores_prompt(review_text):
    return (f"Analyze the following Steam review and score specific aspects of the game. "
            f"Return ONLY a JSON object with keys and values: "
            f"Keys: 'gameplay', 'controls', 'performance', 'bugs', etc. "
            f"Values: numerical score between -1 (worst) and +1 (best). "
            f"Review:\n{review_text}")


def get_feature_requests_prompt(review_text):
    return (f"Consider the following Steam review and identify specific feature requests from the player. "
            f"Return ONLY as JSON with key 'feature_requests' as a list of requested features. "
            f"Review:\n{review_text}")


def get_language_detected_prompt(review_text):
    return (f"Detect the language used in the following Steam game review and provide a normalized English translation. "
            f"Return ONLY a JSON object with keys: "
            f"'language_detected': two-letter language code ('fr', 'en', etc.) and "
            f"'normalized_text_en': the English translation. "
            f"Review:\n{review_text}")


# --- Bonus set prompts ---

def get_quote_highlight_prompt(review_text):
    return (f"Extract a short representative quote from the Steam review (12–20 words). "
            f"Return ONLY a JSON object with key 'quote_highlight'. "
            f"Review:\n{review_text}")


def get_toxicity_prompt(review_text):
    return (f"Assess toxicity/insults/abusive language in the Steam review. "
            f"Return ONLY a JSON object with key 'toxicity_score' as a number between 0 and 1. "
            f"Review:\n{review_text}")


def get_tone_quality_flags_prompt(review_text):
    return (f"Analyze the Steam review for tone/quality flags. Return ONLY a JSON object with keys: "
            f"'sarcasm_flag' (bool), 'humor_flag' (bool), 'spam_flag' (bool), 'coherence_score' (0..1). "
            f"Review:\n{review_text}")


def get_bug_info_prompt(review_text):
    return (f"Determine if the Steam review contains a bug report. Return ONLY a JSON object with keys: "
            f"'bug_report' (bool), 'bug_type' (one of 'crash','freeze','input','save','UI','audio','perf','network','none'), "
            f"and 'steps_hint' (short reproduction hint or empty). "
            f"Review:\n{review_text}")


def get_feature_request_bonus_prompt(review_text):
    return (f"Identify feature requests and a short prioritized suggestion. Return ONLY a JSON object with keys: "
            f"'feature_request' (bool), 'requested_features' (list), and 'suggestion_text' (1–2 sentences). "
            f"Review:\n{review_text}")


def get_nps_and_emotion_prompt(review_text):
    return (f"Classify NPS category and primary emotion for the Steam review. Return ONLY a JSON object with keys: "
            f"'nps_category' (one of 'detractor','passive','promoter') and "
            f"'emotion_primary' (e.g., 'joy','anger','disappointment','excitement','frustration'). "
            f"Review:\n{review_text}")


def get_playtime_bucket_prompt(review_text):
    return (f"Infer the player's playtime bucket from any mentions in the review. "
            f"Return ONLY a JSON object with key 'playtime_bucket' as one of: "
            f"'none','<30min','30–60min','1–5h','5–20h','20h+'. "
            f"If unknown, return 'none'. "
            f"Review:\n{review_text}")


def get_reviewer_experience_prompt(review_text):
    return (f"Infer the reviewer's experience level from the review content. "
            f"Return ONLY a JSON object with key 'reviewer_experience_level' as one of: "
            f"'novice','régulier','core','hardcore'. Use 'régulier' for regular. "
            f"Review:\n{review_text}")


# Pertinence prompt

def get_pertinence_prompt(review_text):
    return (f"Classify whether the Steam review is pertinent for analysis. "
            f"Return ONLY a JSON object with key 'pertinence' (bool). "
            f"Consider 'pertinent' when the review is constructive, informative, and useful for understanding player experience. "
            f"Mark 'non-pertinent' for personal attacks, insults, harassment, or content that is spam/off-topic. "
            f"Review:\n{review_text}")




