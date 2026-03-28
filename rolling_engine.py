"""
PropEdge V9.1 — Rolling Stats Engine
======================================
Live rolling stat computation from game history.
NEVER reads pre-computed L{w}_* columns from CSVs.
ALL windows are computed career-chronologically across seasons.

Rules:
- Use explicit loops or .tail(), NEVER groupby().apply()
- Always datetime sort, never string sort
- Windows require at least 3 games (else NaN)
- Windows L100/L200 require at least half the window
"""
import pandas as pd
import numpy as np
from config import WINDOWS, ROLL_COLS

# Stat columns that exist as raw fields in game log CSVs
RAW_STATS = [
    'MIN_NUM', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
    'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS',
    'WL_WIN', 'WL_LOSS', 'IS_HOME',
    'EFF_FG_PCT', 'TRUE_SHOOTING_PCT', 'USAGE_APPROX', 'FANTASY_PTS',
    'PTS_REB_AST', 'PTS_REB', 'PTS_AST', 'REB_AST',
    'DOUBLE_DOUBLE', 'TRIPLE_DOUBLE',
]


def load_combined(file_2425, file_2526):
    """Load both season CSVs, combine, datetime-sort. Returns DataFrame."""
    df25 = pd.read_csv(file_2425, parse_dates=['GAME_DATE'])
    df26 = pd.read_csv(file_2526, parse_dates=['GAME_DATE'])
    combined = pd.concat([df25, df26], ignore_index=True)
    combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'])
    # CRITICAL: datetime sort, never string sort
    combined = combined.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)
    return combined


def compute_live_rolling(prior_games, window):
    """
    Compute all rolling stat averages for a single window given a DataFrame
    of prior games (already sorted chronologically, most recent last).

    Returns dict of col_name → value for all ROLL_COLS.
    Uses .tail(window) — latest games up to window count.
    Minimum 3 games required; L100/L200 require at least half the window.
    """
    result = {}
    n_games = len(prior_games)
    
    # Determine minimum games required for this window
    if window >= 100:
        min_games = window // 2
    else:
        min_games = min(3, window)
    
    if n_games < min_games:
        for col in ROLL_COLS:
            result[f'L{window}_{col}'] = np.nan
        return result
    
    subset = prior_games.tail(window)
    for col in ROLL_COLS:
        if col not in prior_games.columns:
            result[f'L{window}_{col}'] = np.nan
            continue
        vals = subset[col].dropna()
        if len(vals) < min_games:
            result[f'L{window}_{col}'] = np.nan
        else:
            result[f'L{window}_{col}'] = round(float(vals.mean()), 4)
    
    return result


def get_player_live_stats(player_name, prior_games):
    """
    Given a player's prior game history (sorted chronologically, before prediction date),
    compute ALL rolling windows live.
    
    Returns flat dict: {L3_PTS: x, L5_PTS: x, ..., L200_TRUE_SHOOTING_PCT: x}
    """
    stats = {}
    for w in WINDOWS:
        stats.update(compute_live_rolling(prior_games, w))
    return stats


def build_player_index(combined_df):
    """
    Build a lookup of player_name → sorted game history DataFrame.
    Input DataFrame must already be datetime-sorted.
    
    Returns dict: {player_name: sorted_df}
    """
    pidx = {}
    for pn, grp in combined_df.groupby('PLAYER_NAME'):
        pidx[pn] = grp.sort_values('GAME_DATE').reset_index(drop=True)
    return pidx


def get_prior_games(pidx, player_name, before_date_str):
    """
    Get all games for player strictly before a given date string.
    Returns sorted DataFrame (chronological, oldest first).
    """
    if player_name not in pidx:
        return pd.DataFrame()
    ph = pidx[player_name]
    return ph[ph['GAME_DATE'] < pd.Timestamp(before_date_str)].copy()


def extract_prediction_features(prior, line):
    """
    Given a player's prior games (sorted chronological DataFrame) and a line,
    compute all features needed for prediction:
    - L3/L5/L10/L20/L30 PTS (LIVE, never from CSV columns)
    - rolling FG_PCT, MIN_NUM, FGA (LIVE)
    - std10, volume, trend, fgTrend, minTrend
    - recent20 list, hr10, hr30

    Returns dict of all prediction-ready features.
    """
    if len(prior) < 5:
        return None
    
    # LIVE rolling computation — never read L*_PTS from CSV
    p30 = prior.tail(30); p20 = prior.tail(20)
    p10 = prior.tail(10); p5 = prior.tail(5); p3 = prior.tail(3)
    
    L30 = float(p30['PTS'].mean())
    L20 = float(p20['PTS'].mean())
    L10 = float(p10['PTS'].mean())
    L5  = float(p5['PTS'].mean())
    L3  = float(p3['PTS'].mean())
    
    # FG%
    fg30_raw = p30['FG_PCT'].dropna().mean()
    fg10_raw = p10['FG_PCT'].dropna().mean()
    fg30 = float(fg30_raw) * 100 if pd.notna(fg30_raw) and fg30_raw < 1.5 else float(fg30_raw) if pd.notna(fg30_raw) else None
    fg10 = float(fg10_raw) * 100 if pd.notna(fg10_raw) and fg10_raw < 1.5 else float(fg10_raw) if pd.notna(fg10_raw) else None
    fgTrend = round(fg10 - fg30, 1) if fg30 is not None and fg10 is not None else None
    
    # Minutes
    m30 = float(p30['MIN_NUM'].mean()) if 'MIN_NUM' in prior.columns else None
    m10 = float(p10['MIN_NUM'].mean()) if 'MIN_NUM' in prior.columns else None
    minTrend = round(m10 - m30, 1) if m30 is not None and m10 is not None else None
    
    # FGA
    fga30 = float(p30['FGA'].mean()) if 'FGA' in prior.columns else None
    fga10 = float(p10['FGA'].mean()) if 'FGA' in prior.columns else None
    
    # Std / recent
    std10 = float(p10['PTS'].std()) if len(p10) >= 3 else 5.0
    recent20_pts = list(prior.tail(20)['PTS'].astype(int).values)
    recent10_pts = recent20_pts[-10:]
    r20_homes = list(prior.tail(20)['IS_HOME'].values.astype(int)) if 'IS_HOME' in prior.columns else [0]*len(recent20_pts)
    
    hr10 = round(sum(1 for r in recent10_pts if r > line) / len(recent10_pts) * 100) if recent10_pts else 50
    hr30 = round(sum(1 for r in recent20_pts if r > line) / len(recent20_pts) * 100) if recent20_pts else 50
    
    vol = round(L30 - line, 1)
    trend = round(L5 - L30, 1)
    
    # Minutes model features
    mp_series = prior.tail(10)['MIN_NUM'] if 'MIN_NUM' in prior.columns else pd.Series([30.0]*10)
    min_cv = float(mp_series.std() / mp_series.mean()) if mp_series.mean() > 0 else 1.0
    ppm_vals = (prior.tail(10)['PTS'] / prior.tail(10)['MIN_NUM'].replace(0, np.nan))
    ppm = float(ppm_vals.mean()) if pd.notna(ppm_vals.mean()) else 0.0
    rmt = float(prior.tail(3)['MIN_NUM'].mean() - mp_series.mean()) if 'MIN_NUM' in prior.columns else 0.0
    fpm_vals = (prior.tail(10)['FGA'] / prior.tail(10)['MIN_NUM'].replace(0, np.nan)) if 'FGA' in prior.columns and 'MIN_NUM' in prior.columns else pd.Series([0.0])
    fpm = float(fpm_vals.mean()) if pd.notna(fpm_vals.mean()) else 0.0
    
    return {
        'L30': round(L30, 1), 'L20': round(L20, 1), 'L10': round(L10, 1),
        'L5': round(L5, 1), 'L3': round(L3, 1),
        'fg30': round(fg30, 1) if fg30 is not None else None,
        'fg10': round(fg10, 1) if fg10 is not None else None,
        'fgTrend': fgTrend,
        'fga30': round(fga30, 1) if fga30 is not None else None,
        'fga10': round(fga10, 1) if fga10 is not None else None,
        'm30': round(m30, 1) if m30 is not None else None,
        'm10': round(m10, 1) if m10 is not None else None,
        'minTrend': minTrend,
        'std10': round(std10, 1),
        'vol': vol, 'trend': trend,
        'hr10': hr10, 'hr30': hr30,
        'recent20': recent20_pts,
        'recent10': recent10_pts,
        'r20_homes': r20_homes,
        'min_cv': round(min_cv, 3),
        'ppm': round(ppm, 3),
        'rmt': round(rmt, 1) if pd.notna(rmt) else 0.0,
        'fpm': round(fpm, 3),
    }
