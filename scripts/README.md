# Scripts

Pipeline entry points:
- `scripts/scrape_ncaa_reference.py` — scrape NCAA player tables from Sports Reference
- `scripts/scrape_wnba_reference.py` — scrape WNBA player tables from Basketball Reference
- `scripts/scrape_herhoops.py` — example HerHoopsStats pull (requires login)
- `scripts/build_datasets.py` — build processed NCAA/ WNBA datasets
- `scripts/build_model_frame.py` — merge datasets into `model_df.csv`
- `scripts/train_model.py` — train baseline classifier and save artifacts
