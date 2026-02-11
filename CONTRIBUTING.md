# Contributing

Thanks for your interest in contributing.

## Workflow
1. Open an issue describing the change.
2. Create a branch from `main`.
3. Keep changes focused and well-documented.
4. Include tests or validation notes where possible.

## Code Style
- Prefer small, composable functions.
- Avoid hard-coded paths; use `ncaaw_stats.config`.
- Keep data out of version control unless it lives in `data/samples/`.

## Running the Pipeline
```
python scripts/build_datasets.py
python scripts/build_model_frame.py
python scripts/train_model.py
```

## Reporting Bugs
Please include:
- Steps to reproduce
- Expected vs actual behavior
- Any relevant logs or stack traces
