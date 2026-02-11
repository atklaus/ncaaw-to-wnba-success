# Architecture

## High-Level Flow
```
NCAA/HerHoops/Reference data
        │
        ▼
  data/raw/ ── build_datasets.py ──► data/processed/
        │                             │
        ▼                             ▼
  build_model_frame.py           model_df.csv
        │                             │
        ▼                             ▼
  train_model.py  ───────────────► artifacts/ + reports/figures/
```

## Design Principles
- Separate raw data, processed data, and artifacts for clarity.
- Keep scraping scripts isolated from modeling logic.
- Use deterministic preprocessing steps so results can be reproduced.

## Module Responsibilities
- `ncaaw_stats.data`: IO and dataset transformations
- `ncaaw_stats.scrapers`: source-specific scraping routines
- `ncaaw_stats.modeling`: feature prep and training
- `scripts/`: CLI entry points for pipeline stages
