# NCAA Women's Basketball -> WNBA Success Pipeline

A clean, end-to-end data pipeline that ingests NCAA women's basketball stats, joins them with WNBA outcomes, and trains a baseline model to identify which collegiate profiles translate to professional success.

## Overview
This project turns a messy research workflow into a reproducible, open-source pipeline:
- Scrape or ingest NCAA and WNBA player statistics
- Standardize and merge datasets into a modeling-ready table
- Train and evaluate a baseline classifier
- Publish figures and artifacts for analysis and communication

## The Problem It Solves
Women’s basketball data lives across multiple sources and formats. This repo gives you a repeatable way to collect, clean, and analyze NCAA player performance and its relationship to WNBA outcomes (e.g., win shares per 48 minutes).

## Architecture
```
raw data (scraped) ─▶ data/raw/
                  └▶ build_datasets.py ─▶ data/processed/
                                      └▶ build_model_frame.py ─▶ data/processed/model_df.csv
                                                                  └▶ train_model.py ─▶ artifacts/
                                                                                   └▶ reports/figures/
```

## Tech Stack
- Python 3.10+
- pandas, numpy
- requests, BeautifulSoup, lxml, html5lib (scraping)
- scikit-learn, joblib (modeling)
- matplotlib, seaborn (optional visualization)

## Setup
```
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .[dev,viz]
```

If you prefer, use the Makefile:
```
make setup
```

## Usage
### 1. Scrape or ingest raw data
- NCAA player URLs are expected in `data/raw/ncaa_player_urls.csv` with `player_name` and `sports_reference_url` columns.
- WNBA data is scraped alphabetically by player last name.

```
python scripts/scrape_ncaa_reference.py
python scripts/scrape_wnba_reference.py
```

### 2. Build processed datasets
```
python scripts/build_datasets.py
```
Outputs:
- `data/processed/ncaa_processed.csv`
- `data/processed/wnba_processed.csv`

### 3. Build the modeling frame
```
python scripts/build_model_frame.py
```
Output:
- `data/processed/model_df.csv`

### 4. Train the baseline model
```
python scripts/train_model.py
```
Outputs:
- `artifacts/wnba_success_model.joblib`
- `artifacts/training_metrics.csv`
- `artifacts/model_features.csv`

## Screenshots / UI
The latest exploratory charts are stored in `reports/figures/`.

## Repository Structure
```
.
├── artifacts/            # Trained models and metrics
├── data/
│   ├── raw/              # Raw scraped data (not committed)
│   ├── processed/        # Clean datasets (not committed)
│   └── samples/          # Small sample datasets committed to git
├── docs/                 # Architecture and data documentation
├── reports/              # Figures and slides
├── scripts/              # CLI scripts for pipeline steps
└── src/ncaaw_stats/       # Core library code
```

## Notes on Data Access
- HerHoopsStats requires login credentials. Provide `HERHOOPS_USERNAME` and `HERHOOPS_PASSWORD` in a `.env` file (see `.env.example`).
- Some source datasets may have licensing or rate-limiting restrictions. Be respectful and cache results locally.

## Contributing
See `docs/README.md` for architecture and data dictionary notes before opening a PR.
