# Pipeline Guide

## 1. Scrape/Collect Raw Data
- NCAA: provide `data/raw/ncaa_player_urls.csv` with a `sports_reference_url` per player.
- WNBA: use `scripts/scrape_wnba_reference.py` to crawl Basketball Reference by letter.

Outputs:
- `data/raw/ncaa_ref/*.csv`
- `data/raw/wnba_ref/*.csv`

## 2. Build Processed Datasets
```
python scripts/build_datasets.py
```
Outputs:
- `data/processed/ncaa_processed.csv`
- `data/processed/wnba_processed.csv`

## 3. Build Modeling Frame
```
python scripts/build_model_frame.py
```
Output:
- `data/processed/model_df.csv`

## 4. Train Model
```
python scripts/train_model.py
```
Outputs:
- `artifacts/wnba_success_model.joblib`
- `artifacts/training_metrics.csv`
- `artifacts/model_features.csv`
