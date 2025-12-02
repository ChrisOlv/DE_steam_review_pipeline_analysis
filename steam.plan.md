<!-- 93e4e3e7-eed8-4891-9216-8fe84448eace 407bcaa3-147c-4e45-bfab-73ddbc936b49 -->
# Steam Reviews Analytics – targeted enhancements

## Scope and constraints

- Audience: product/dev triage and data exploration for a single game
- Data volume: <1k rows (in-memory Pandas OK)
- Goal: overview → filter → drilldown to actionable insights

## 1) Quick wins (UI and filters)

- Add date range filter and language multi-select in sidebar.
- Add sentiment and toxicity sliders; boolean toggles for `llm_bug_reported_flag`, `llm_feature_requested_flag`, `llm_review_pertinence_flag`, `llm_spam_flag`.
- Add text search box across `review` and `llm_review_translated`.
- Show total filtered rows and share/export buttons (CSV + JSONL).

## 2) Thematic insights (cards + charts)

- Top themes/keywords: bar chart using `llm_themes` and `llm_keywords` (explode lists).
- Emotion distribution: keep current, add donut with percentage.
- Aspect radar: parse `llm_score` JSON to radar chart (gameplay/controls/performance/...); show mean and IQR whiskers via small tooltip.
- NPS and recommendation: segmented bar for `llm_NPS` and % recommended over time (already partially present).

## 3) Review explorer (drilldown panel)

- Replace `st.data_editor` with a table that supports row selection.
- On select: right-side expander shows
- Original vs translated review (toggle), `quote_highlight`, `tl_dr`.
- Chips for `llm_themes`, `llm_keywords`, pros/cons.
- Badges: toxicity, sarcasm, humor, spam.
- Metadata: playtime, experience level, dates, votes.

## 4) Bug triage view

- New Tab: “Bugs” filtered on `llm_bug_reported_flag=true`.
- Group by `bug_type`; table with `llm_bug_report_text`, `quote_highlight`, count, last seen date.
- Small trend line by bug_type over time.

## 5) Feature requests board

- New Tab: “Feature requests” filtered on `llm_feature_requested_flag=true`.
- Group by `llm_feature_requested_tag`; show example `llm_feature_requested_text` and counts.
- Copy-to-clipboard list for top tags to paste into backlog.

## 6) Cohorts and balance signals

- Cohorts by `playtime_bucket` and `reviewer_experience_level` → recommend rate and avg aspect scores.
- Difficulty/balance signal: percent of reviews mentioning balance via theme keyword match + sentiment delta.

## 7) Data quality and coverage

- Small panel: translation coverage, language mix, null rates for key LLM fields, last update date (from `date_updated`).

## 8) Performance/caching

- Keep full-load cache; apply all filters client-side (given <1k rows).
- Factor filter logic into a function; ensure Altair datasets are small and typed.

## Key code touches

- `streamlit_app.py`
- Sidebar: new controls (date range, language, sliders, search).
- Helpers: `explode_list_columns`, `parse_llm_score`, `apply_filters`.
- Layout: tabs → Overview | Explorer | Bugs | Feature requests | Cohorts | Quality.
- Charts: Altair bar/donut/radar (use normalized scores 0–1).

## Out-of-scope (for later)

- Server-side SQL filtering (not needed for <1k rows)
- Auth/role management
- Real-time alerts

### To-dos

- [ ] Add date range, language selects, sentiment/toxicity sliders, text search
- [ ] Top themes/keywords bars; emotion donut; aspect radar from llm_score JSON
- [ ] Implement row-select drilldown with original/translated, chips, badges
- [ ] Create Bugs tab grouped by bug_type with examples and trends
- [ ] Create Feature requests tab grouped by requested tag with copy-out list
- [ ] Add cohort charts by playtime and experience; balance signal metric
- [ ] Add data quality/coverage panel and last update date
- [ ] Factor filtering helpers; ensure caching and types for charts


