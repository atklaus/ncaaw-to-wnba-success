# Data Dictionary (Core Fields)

## `data/processed/ncaa_processed.csv`
- `name`: Player name inferred from filename
- `pg_school`: College program
- `pg_season`: Season label (`YYYY-YY` or `Career`)
- `tot_pts`: Career total points
- `conference`: Conference derived from per-game table
- `last_season`: Last NCAA season year detected

## `data/processed/wnba_processed.csv`
- `player_name`: WNBA player name
- `college_team`: NCAA college name
- `adv_ws_48`: WNBA win shares per 48 minutes
- `adv_per`: WNBA player efficiency rating
- `debut_year`: First WNBA season

## `data/processed/model_df.csv`
- `ws_48_pro`: Target label (derived from `adv_ws_48`)
- `per_pro`: WNBA PER (optional model feature)
- `college_team`: Standardized college program
- `conference`: NCAA conference
- `position`: NCAA position (if available)
- Additional NCAA stats columns from Sports Reference
