# Steam Review Analysis Pipeline

This repository contains a pipeline for ingesting, enriching, and analyzing Steam game reviews using Azure OpenAI and DuckDB. The pipeline is orchestrated using GitHub Actions to automate the process every hour.

## Table of Contents

- [Directory Structure](#directory-structure)
- [How It Works](#how-it-works)
- [Setup Instructions](#setup-instructions)
- [Environment Variables](#environment-variables)
- [GitHub Actions](#github-actions)
- [Contributing](#contributing)
- [License](#license)

## Directory Structure

```
.
├── .github
│   └── workflows
│       └── llm_enrichment.yml  # GitHub Actions workflow for ingestion and enrichment
├── dataviz                      # Directory for storing exported Parquet files
├── export_dataviz.py            # Script to export data from DuckDB to Parquet files
├── ingest_steam.py              # Script to ingest Steam reviews into DuckDB
├── enrich_sentiment.py          # Script to enrich reviews with sentiment analysis
├── prompts.py                   # Contains prompt templates for Azure OpenAI
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## How It Works

1. **Ingestion**: The `ingest_steam.py` script fetches the latest Steam reviews using the Steam API and stores them in the `raw_reviews` table in DuckDB. It tracks the last ingested review timestamp to avoid duplicates.

2. **Enrichment**: The `enrich_sentiment.py` script processes the reviews in `raw_reviews`, performing sentiment analysis and generating additional insights using Azure OpenAI. The enriched data is stored in the `llm_enrichment` table.

3. **Export**: The `export_dataviz.py` script exports the data from the `ingest_state`, `llm_enrichment`, and `raw_reviews` tables into Parquet files for further analysis and visualization.

4. **Automation**: The entire process is orchestrated using GitHub Actions, which runs the ingestion, enrichment, and export processes every four hours.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ChrisOlv/DE_steam_review_pipeline_analysis.git
   cd <repository-name>
   Create a venv
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.11 installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the root directory with the following variables (for new dev purposes only)
   ```plaintext
   MOTHERDUCK_TOKEN=<your_motherduck_token>
   AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
   AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>
   AZURE_OPENAI_DEPLOYMENT=<your_azure_openai_deployment>
   AZURE_OPENAI_API_VERSION=<your_azure_openai_api_version>  # Optional
   MD_DB_NAME=<your_database_name>  # Optional, defaults to steam_analytics
   ```

## dataviz
Dashboard is hosted on Streamlit : https://analytics-deathliver.streamlit.app/


## GitHub Actions

The GitHub Actions workflow is defined in `.github/workflows/llm_enrichment.yml`. It includes two jobs:
- **Ingest Steam Data**: Runs `steam_ingest.py` to fetch and store new reviews.
- **Run Enrichment**: Runs `enrich_sentiment.py` to analyze and enrich the reviews, followed by exporting the results to Parquet files.

The workflow is scheduled to run every 1hour and can also be triggered manually.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
