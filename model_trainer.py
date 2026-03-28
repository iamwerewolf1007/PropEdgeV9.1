"""
PropEdge V9.1 — Projection Model Trainer
==========================================
Uses LIVE rolling stats (never reads pre-computed L*_PTS CSV columns).
Career-chronological: both seasons concatenated and sorted by date before rolling.

Performance fixes:
- All rolling columns built via pd.concat (no fragmented frame inserts)
- GBR uses n_iter_no_change=10 for early stopping (prevents hang on 43k rows)
- Matches old training row count exactly: min 10 prior games per row
"""
import pandas as pd
import numpy as np
import pickle, json
from sklearn.ensemble import GradientBoostingRegressor
from config import get_dvp, POS_MAP

FEATURES = ['l30','l10','l5','l3','volume','trend','std10','defP','pace_rank',
            'h2h_ts_dev','h2h_fga_dev','h2h_min_dev','h2h_conf',
            'min_cv','pts_per_min','recent_min_trend','fga_per_min',
            'is_b2b','rest_days','consistency','line']


def _roll_shift(series, window):
    """Rolling mean of prior `window` games only (shift(1) = no lookahead)."""
    return series.rolling(window, min_periods=1).mean().shift(1)


def build_training_data(file_2425, file_2526, file_h2h):
    """
    Build training samples using vectorised rolling. Matches old row count
    exactly: only rows with >= 10 prior games (old code: range(10, len(grp))).
    All temp columns built via pd.concat to avoid PerformanceWarning.
    """
    df25 = pd.read_csv(file_2425, parse_dates=['GAME_DATE'])
    df26 = pd.read_csv(file_2526, parse_dates=['GAME_DATE'])
    h2h  = pd.read_csv(file_h2h)

    # CRITICAL: datetime sort across both seasons — never string sort
    combined = pd.concat([df25, df26], ignore_index=True)
    combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'])
    combined = combined.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)

    h2h_lkp = {(r['PLAYER_NAME'], r['OPPONENT']): r for _, r in h2h.iterrows()}
    print(f"    Rows: {len(combined):,}  Players: {combined['PLAYER_NAME'].nunique():,}")

    # ── Build all rolling columns in one pd.concat — no fragmentation ─────────
    # groupby preserves sort order within each player group
    grp = combined.groupby('PLAYER_NAME', sort=False)

    roll_cols = pd.concat([
        grp['PTS'].transform(lambda s: _roll_shift(s, 30)).rename('_l30'),
        grp['PTS'].transform(lambda s: _roll_shift(s, 10)).rename('_l10'),
        grp['PTS'].transform(lambda s: _roll_shift(s, 5) ).rename('_l5'),
        grp['PTS'].transform(lambda s: _roll_shift(s, 3) ).rename('_l3'),
        grp['PTS'].transform(
            lambda s: s.rolling(10, min_periods=3).std().shift(1)
        ).fillna(5.0).rename('_std10'),
        grp['MIN_NUM'].transform(lambda s: _roll_shift(s, 10)).rename('_m10'),
        grp['MIN_NUM'].transform(lambda s: _roll_shift(s, 3) ).rename('_m3'),
        grp['FGA'].transform(   lambda s: _roll_shift(s, 10)).rename('_fga10'),
        grp['GAME_DATE'].transform(
            lambda s: s.diff().dt.days.fillna(99)
        ).rename('_rest'),
        # Game count per player (cumcount starts at 0 = 1st game)
        grp['GAME_DATE'].transform('cumcount').rename('_game_num'),
    ], axis=1)

    # Only keep the columns we need from combined (avoid fragmentation on join)
    base = combined[['PLAYER_NAME', 'GAME_DATE', 'PTS', 'OPPONENT',
                      'PLAYER_POSITION']].copy()
    base = pd.concat([base, roll_cols], axis=1)

    # Filter: require at least 10 prior games (matches old range(10, len(grp)))
    # _game_num == 10 means this is the 11th game, with 10 prior games available
    base = base[base['_game_num'] >= 10].dropna(subset=['_l30']).copy()
    print(f"    Training rows after filters: {len(base):,}")

    # ── Build feature rows (vectorised, no Python loop) ───────────────────────
    base['_l30']   = base['_l30'].astype(float)
    base['_l10']   = base['_l10'].astype(float)
    base['_l5']    = base['_l5'].astype(float)
    base['_l3']    = base['_l3'].astype(float)
    base['_std10'] = base['_std10'].astype(float)
    base['_m10']   = base['_m10'].fillna(28.0).astype(float)
    base['_m3']    = base['_m3'].fillna(28.0).astype(float)
    base['_fga10'] = base['_fga10'].fillna(8.0).astype(float)
    base['_rest']  = base['_rest'].astype(float)

    # Synthetic line = round L30 to nearest 0.5
    base['line']      = (base['_l30'] * 2).round() / 2
    base['volume']    = (base['_l30'] - base['line']).round(1)
    base['trend']     = (base['_l5']  - base['_l30']).round(1)
    base['min_cv']    = (base['_std10'] / base['_m10'].clip(lower=1)).round(3)
    base['pts_per_min']      = (base['_l10'] / base['_m10'].clip(lower=1)).round(3)
    base['recent_min_trend'] = (base['_m3'] - base['_m10']).round(1)
    base['fga_per_min']      = (base['_fga10'] / base['_m10'].clip(lower=1)).round(3)
    base['is_b2b']    = (base['_rest'] == 1).astype(int)
    base['rest_days'] = base['_rest'].astype(int)
    base['consistency'] = (1 / (base['_std10'] + 1)).round(3)

    # DVP and H2H (require Python lookup — vectorise where possible)
    def get_dvp_row(row):
        pos = POS_MAP.get(str(row['PLAYER_POSITION']), 'Forward')
        return get_dvp(row['OPPONENT'], pos)

    base['defP'] = base.apply(get_dvp_row, axis=1)
    base['pace_rank'] = 15  # placeholder; pace computed at predict time

    def get_h2h(row):
        hr = h2h_lkp.get((row['PLAYER_NAME'], row['OPPONENT']))
        if hr is None:
            return 0, 0, 0, 0
        ts   = float(hr['H2H_TS_VS_OVERALL'])   if pd.notna(hr.get('H2H_TS_VS_OVERALL'))   else 0
        fga  = float(hr['H2H_FGA_VS_OVERALL'])  if pd.notna(hr.get('H2H_FGA_VS_OVERALL'))  else 0
        mn   = float(hr['H2H_MIN_VS_OVERALL'])  if pd.notna(hr.get('H2H_MIN_VS_OVERALL'))  else 0
        conf = float(hr['H2H_CONFIDENCE'])      if pd.notna(hr.get('H2H_CONFIDENCE'))      else 0
        return ts, fga, mn, conf

    h2h_vals = base.apply(get_h2h, axis=1, result_type='expand')
    h2h_vals.columns = ['h2h_ts_dev', 'h2h_fga_dev', 'h2h_min_dev', 'h2h_conf']
    base = pd.concat([base, h2h_vals], axis=1)

    # Rename to feature names
    base = base.rename(columns={
        '_l30': 'l30', '_l10': 'l10', '_l5': 'l5', '_l3': 'l3', '_std10': 'std10'
    })
    base['actual_pts'] = combined.loc[base.index, 'PTS'].astype(int)

    print(f"    Training samples: {len(base):,}")
    return base


def train_and_save(file_2425, file_2526, file_h2h, model_file, trust_file):
    """Train GBR model with early stopping, save model + player trust scores."""
    print("    Building training data (vectorised rolling)...")
    train_df = build_training_data(file_2425, file_2526, file_h2h)

    X = train_df[FEATURES].fillna(0)
    y = train_df['actual_pts']

    # n_iter_no_change=15: stops early if validation loss doesn't improve
    # validation_fraction=0.1: holds out 10% for early stopping check
    # This prevents the hang on large datasets while preserving accuracy
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,               # stochastic GBR — faster + less overfit
        n_iter_no_change=15,         # early stopping
        validation_fraction=0.1,
        tol=1e-4,
        random_state=42,
    )
    model.fit(X, y)
    actual_trees = model.n_estimators_
    print(f"    GBR: {actual_trees} trees (early stop from max 300)")

    model_file.parent.mkdir(parents=True, exist_ok=True)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"    ✓ Model → {model_file.name}")

    # Player trust scores
    train_df['pred'] = model.predict(X)
    train_df['correct'] = (
        ((train_df['pred'] > train_df['line']) & (train_df['actual_pts'] > train_df['line'])) |
        ((train_df['pred'] < train_df['line']) & (train_df['actual_pts'] < train_df['line']))
    ).astype(int)

    trust = {
        p: round(float(g['correct'].mean()), 3)
        for p, g in train_df.groupby('PLAYER_NAME')
        if len(g) >= 10
    }
    with open(trust_file, 'w') as f:
        json.dump(trust, f, indent=2)
    print(f"    ✓ Trust: {len(trust)} players → {trust_file.name}")
    print(f"    In-sample accuracy: {float(train_df['correct'].mean()):.1%}")
    return model
