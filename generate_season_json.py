#!/usr/bin/env python3
"""
PropEdge V9.1 — Generate Season JSONs
========================================
Fixes:
- Rolling stats computed LIVE (never from stale CSV columns)
- Unique pre/post-match reasoning via reasoning_engine.py

1. Generate season_2024_25.json with synthetic lines + model predictions + grades
2. Generate season_2025_26.json with real prop lines + grades
Both files LOCKED after generation.
"""
import pandas as pd
import numpy as np
import json, sys, time, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import *
from synthetic_lines import generate_season_lines
from audit import log_event, log_batch_summary
from rolling_engine import load_combined, build_player_index, get_prior_games, extract_prediction_features
from reasoning_engine import generate_pre_match_reason, generate_post_match_reason


def _safe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 4)
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, pd.Timestamp): return v.strftime('%Y-%m-%d')
    return v


def _clean_for_json(obj):
    if isinstance(obj, dict):   return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):  return int(obj)
    elif isinstance(obj, (np.floating,)): return None if np.isnan(obj) else round(float(obj), 4)
    elif isinstance(obj, np.bool_):       return bool(obj)
    elif isinstance(obj, pd.Timestamp):   return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, np.ndarray):     return [_clean_for_json(v) for v in obj.tolist()]
    elif isinstance(obj, float) and np.isnan(obj): return None
    return obj


def run_model_on_props(props_df, all_logs, h2h_df, model, player_trust, season_label):
    """
    Run full prediction engine on a props DataFrame.
    Rolling stats are LIVE-computed from game history — never reading CSV columns.
    Returns list of play dicts.
    """
    # Build indexes from combined game logs
    all_logs = all_logs.copy()
    all_logs['GAME_DATE'] = pd.to_datetime(all_logs['GAME_DATE'])
    # CRITICAL: datetime sort, never string sort
    all_logs = all_logs.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)

    h2h_lkp = {(r['PLAYER_NAME'], r['OPPONENT']): r for _, r in h2h_df.iterrows()}
    pidx = build_player_index(all_logs)

    # Pace rank
    team_fga = all_logs.groupby('OPPONENT')['FGA'].mean()
    pace_rank = {t: i + 1 for i, (t, _) in enumerate(team_fga.sort_values(ascending=False).items())}

    # B2B map
    b2b_map = {}
    for pn, g in all_logs.sort_values('GAME_DATE').groupby('PLAYER_NAME'):
        dates = g['GAME_DATE'].values
        for i in range(len(dates)):
            ds = pd.Timestamp(dates[i]).strftime('%Y-%m-%d')
            b2b_map[(pn, ds)] = int((dates[i] - dates[i-1]).astype('timedelta64[D]').astype(int)) if i > 0 else 99

    # Actual results lookup
    results_lkp = {}
    for _, r in all_logs.iterrows():
        results_lkp[(r['PLAYER_NAME'], r['GAME_DATE'].strftime('%Y-%m-%d'))] = r['PTS']

    plays = []; skipped = 0; processed = 0
    total = len(props_df)

    for _, prop in props_df.iterrows():
        processed += 1
        if processed % 2000 == 0: print(f"    {processed}/{total}...")

        player    = prop['Player']
        date_str  = prop['Date'].strftime('%Y-%m-%d')
        line      = prop['Line']
        game      = prop['Game']
        home_team = prop['Home']
        away_team = prop['Away']
        raw_pos   = str(prop.get('Position', '') or '')
        position  = POS_MAP.get(raw_pos, 'Forward')

        if player not in pidx: skipped += 1; continue

        # LIVE rolling stats — never read L*_PTS from CSV
        prior = get_prior_games(pidx, player, date_str)
        if len(prior) < 5: skipped += 1; continue

        feats = extract_prediction_features(prior, line)
        if feats is None: skipped += 1; continue

        L30 = feats['L30']; L20 = feats['L20']; L10 = feats['L10']
        L5  = feats['L5'];  L3  = feats['L3']
        vol = feats['vol']; trend = feats['trend']; std10 = feats['std10']
        hr10 = feats['hr10']; hr30 = feats['hr30']
        r20 = feats['recent20']; r20h = feats['r20_homes']
        fg30 = feats['fg30']; fg10 = feats['fg10']; fgTrend = feats['fgTrend']
        m30  = feats['m30'];  m10  = feats['m10'];  minTrend = feats['minTrend']
        fga30 = feats['fga30']; fga10 = feats['fga10']
        min_cv = feats['min_cv']; ppm = feats['ppm']; rmt = feats['rmt']; fpm = feats['fpm']

        # Team/opponent/position from latest prior game
        sn = prior.iloc[-1]
        team_abr = sn.get('GAME_TEAM_ABBREVIATION', '')
        is_home  = team_abr == home_team
        opp      = away_team if is_home else home_team

        # H2H / defence
        hr_  = h2h_lkp.get((player, opp))
        h2hG = int(hr_['H2H_GAMES'])           if hr_ is not None else 0
        h2h_avg  = float(hr_['H2H_AVG_PTS'])   if hr_ is not None else None
        h2h_ts   = float(hr_['H2H_TS_VS_OVERALL']) if hr_ is not None and pd.notna(hr_.get('H2H_TS_VS_OVERALL')) else 0
        h2h_fga  = float(hr_['H2H_FGA_VS_OVERALL'])if hr_ is not None and pd.notna(hr_.get('H2H_FGA_VS_OVERALL'))else 0
        h2h_min  = float(hr_['H2H_MIN_VS_OVERALL'])if hr_ is not None and pd.notna(hr_.get('H2H_MIN_VS_OVERALL'))else 0
        h2h_conf = float(hr_['H2H_CONFIDENCE']) if hr_ is not None and pd.notna(hr_.get('H2H_CONFIDENCE'))    else 0
        h2h_str  = f"{h2h_avg:.1f} ({h2hG}g)"  if h2hG >= 3 and h2h_avg else ""
        use_h2h  = h2hG >= 3 and h2h_avg is not None

        defP  = get_dvp(opp, position); defO = get_def_overall(opp)
        op    = pace_rank.get(opp, 15)

        rest  = b2b_map.get((player, date_str), 99); is_b2b = 1 if rest == 1 else 0

        # ── Projection model ──
        pred_pts = None; pred_gap = 0
        if model is not None:
            from model_trainer import FEATURES
            fd = {
                'l30': L30, 'l10': L10, 'l5': L5, 'l3': L3,
                'volume': vol, 'trend': trend, 'std10': std10,
                'defP': defP, 'pace_rank': op,
                'h2h_ts_dev': h2h_ts, 'h2h_fga_dev': h2h_fga,
                'h2h_min_dev': h2h_min, 'h2h_conf': h2h_conf,
                'min_cv': min_cv,
                'pts_per_min': ppm if pd.notna(ppm) else 0,
                'recent_min_trend': rmt if pd.notna(rmt) else 0,
                'fga_per_min': fpm if pd.notna(fpm) else 0,
                'is_b2b': is_b2b, 'rest_days': rest,
                'consistency': 1 / (std10 + 1), 'line': line,
            }
            Xp = pd.DataFrame([fd])[FEATURES].fillna(0)
            pred_pts = float(model.predict(Xp)[0]); pred_gap = abs(pred_pts - line)

        # ── 10-signal model ──
        W = POS_WEIGHTS.get(position, POS_WEIGHTS['Forward'])
        s = {
            1: np.clip((L30 - line) / 5, -1, 1),
            2: (hr30 / 100 - 0.5) * 2,
            3: (hr10 / 100 - 0.5) * 2,
            4: np.clip((L5 - L30) / 5, -1, 1),
            5: np.clip(vol / 5, -1, 1),
            6: np.clip((defP - 15) / 15, -1, 1),
            7: np.clip((h2h_avg - line) / 5, -1, 1) if use_h2h else 0.0,
            8: np.clip((15 - op) / 15, -1, 1),
            9: np.clip((fgTrend or 0) / 10, -1, 1),
            10: np.clip((minTrend or 0) / 5, -1, 1),
        }
        if not use_h2h:
            tw = sum(w for k, w in W.items() if k != 7)
            ws = sum(W[k] * s[k] for k in s if k != 7)
        else:
            tw = sum(W.values()); ws = sum(W[k] * s[k] for k in s)
        composite = ws / tw

        # ── Direction ──
        if pred_pts is not None:
            if pred_pts > line + 0.3:   direction = 'OVER';  is_lean = False
            elif pred_pts < line - 0.3: direction = 'UNDER'; is_lean = False
            else:
                raw_dir = 'OVER' if pred_pts > line else 'UNDER'
                direction = f'LEAN {raw_dir}'; is_lean = True
        else:
            if composite > 0.05:   direction = 'OVER';  is_lean = False
            elif composite < -0.05: direction = 'UNDER'; is_lean = False
            else:
                raw_dir = 'OVER' if composite > 0 else 'UNDER'
                direction = f'LEAN {raw_dir}'; is_lean = True

        # ── Confidence ──
        sig_conf  = float(np.clip(0.5 + abs(composite) * 0.3, 0.50, 0.85))
        if std10 > 8: sig_conf -= 0.03
        sig_conf  = float(np.clip(sig_conf, 0.45, 0.85))
        proj_conf = float(np.clip(0.5 + pred_gap * 0.04, 0.45, 0.90)) if pred_pts else sig_conf
        conf = 0.4 * sig_conf + 0.6 * proj_conf

        # ── Flags ──
        is_over = 'UNDER' not in direction
        flags = 0; flag_details = []
        for nm, ag, dt in [
            ('Volume',   (is_over and vol > 0) or (not is_over and vol < 0),                  f"{vol:+.1f}"),
            ('HR L30',   (is_over and hr30 > 50) or (not is_over and hr30 < 50),              f"{hr30}%"),
            ('HR L10',   (is_over and hr10 > 50) or (not is_over and hr10 < 50),              f"{hr10}%"),
            ('Trend',    (is_over and trend > 0) or (not is_over and trend < 0),              f"{trend:+.1f}"),
            ('Context',  (is_over and vol > -1) or (not is_over and vol < 1),                 f"vol={vol:+.1f}"),
            ('Defense',  (is_over and defP > 15) or (not is_over and defP < 15),             f"#{defP}"),
            ('H2H',      use_h2h and ((is_over and h2h_avg > line) or (not is_over and h2h_avg < line)),
                         f"{h2h_avg:.1f}" if use_h2h else "N/A"),
            ('Pace',     (is_over and op < 15) or (not is_over and op > 15),                 f"#{op}"),
            ('FG Trend', fgTrend is not None and ((is_over and fgTrend > 0) or (not is_over and fgTrend < 0)),
                         f"{fgTrend:+.1f}%" if fgTrend else "N/A"),
            ('Min Trend',minTrend is not None and ((is_over and minTrend > 0) or (not is_over and minTrend < 0)),
                         f"{minTrend:+.1f}" if minTrend else "N/A"),
        ]:
            flags += 1 if ag else 0
            flag_details.append({'name': nm, 'agrees': bool(ag), 'detail': dt})

        h2h_aligned = True
        if h2h_ts != 0:
            if 'OVER' in direction and h2h_ts < -3:  h2h_aligned = False
            elif 'UNDER' in direction and h2h_ts > 3: h2h_aligned = False

        # ── Tier ──
        if is_lean:
            tier = 3; tl = 'T3_LEAN'
        elif conf >= 0.70 and flags >= 8 and std10 <= 6 and h2h_aligned: tier = 1; tl = 'T1_ULTRA'
        elif conf >= 0.65 and flags >= 7 and std10 <= 7 and h2h_aligned: tier = 1; tl = 'T1_PREMIUM'
        elif conf >= 0.62 and flags >= 7 and std10 <= 7 and h2h_aligned: tier = 1; tl = 'T1'
        elif conf >= 0.55 and flags >= 6 and std10 <= 8 and h2h_aligned: tier = 2; tl = 'T2'
        else:                                                               tier = 3; tl = 'T3'

        tr = player_trust.get(player)
        if tr is not None and tr < 0.42 and tier == 1: tier = 2; tl = 'T2'
        units = 3.0 if tl == 'T1_ULTRA' else 2.0 if tier == 1 else 1.0 if tier == 2 else 0.0

        # ── Grade ──
        actual = results_lkp.get((player, date_str))
        if actual is None and 'Actual_PTS' in prop and pd.notna(prop.get('Actual_PTS')):
            actual = int(prop['Actual_PTS'])
        if actual is not None and not (isinstance(actual, float) and np.isnan(actual)):
            actual = int(actual)
            if 'OVER' in direction:   result = 'WIN' if actual > line else 'LOSS'
            elif 'UNDER' in direction: result = 'WIN' if actual < line else 'LOSS'
            else:                      result = 'NO PLAY'
            delta = round(actual - line, 1)
        else:
            actual = None; result = None; delta = None

        over_odds  = american_to_decimal(prop.get('Over Odds'))
        under_odds = american_to_decimal(prop.get('Under Odds'))
        ro = sum(1 for r in r20 if r > line)
        ru = sum(1 for r in r20 if r <= line)

        # ── Reasoning ──
        play_preview = {
            'player': player, 'dir': direction, 'line': line,
            'l30': L30, 'l10': L10, 'l5': L5, 'l3': L3,
            'volume': vol, 'trend': trend, 'std10': std10,
            'flags': flags, 'flagDetails': flag_details,
            'h2h': h2h_str, 'h2hG': h2hG, 'h2hTsDev': h2h_ts,
            'h2hFgaDev': h2h_fga,
            'h2hProfile': hr_.get('H2H_SCORING_PROFILE', '') if hr_ is not None else '',
            'defP': defP, 'defO': defO, 'pace': op,
            'fgTrend': fgTrend, 'minTrend': minTrend,
            'minL30': m30, 'minL10': m10, 'conf': conf,
            'predPts': round(pred_pts, 1) if pred_pts else None,
            'predGap': round(pred_gap, 1) if pred_pts else None,
            'tierLabel': tl, 'position': position,
            'match': game, 'isHome': is_home,
            'recent': r20[:5], 'hr30': hr30, 'hr10': hr10,
        }
        pre_reason = generate_pre_match_reason(play_preview)

        # Post-match reasoning if graded
        post_reason = ''; loss_type = None
        if result in ('WIN', 'LOSS'):
            play_for_post = {
                **play_preview,
                'actualPts': actual, 'result': result, 'delta': delta,
            }
            post_reason, loss_type = generate_post_match_reason(play_for_post)

        plays.append({
            'date': date_str, 'player': player, 'match': game, 'fullMatch': game,
            'isHome': is_home, 'team': str(team_abr), 'gameTime': '', 'position': position,
            'posSimple': position[:1], 'line': _safe(line),
            'overOdds': _safe(over_odds), 'underOdds': _safe(under_odds),
            'books': _safe(prop.get('Books', 1)), 'minLine': None, 'maxLine': None,
            'spread': None, 'total': None, 'blowout': False,
            'l30': _safe(round(float(L30), 1)), 'l20': _safe(round(float(L20), 1)),
            'l10': _safe(round(float(L10), 1)), 'l5':  _safe(round(float(L5), 1)),
            'l3':  _safe(round(float(L3), 1)),
            'hr30': hr30, 'hr10': hr10,
            'recent': r20[:5], 'recent10': r20[:10], 'recent20': r20,
            'recent20homes': [bool(x) for x in r20h],
            'defO': defO, 'defP': defP, 'pace': op,
            'h2h': h2h_str, 'h2hG': h2hG,
            'h2hTsDev': _safe(h2h_ts), 'h2hFgaDev': _safe(h2h_fga),
            'h2hConfidence': _safe(h2h_conf),
            'h2hProfile': hr_.get('H2H_SCORING_PROFILE', '') if hr_ is not None else '',
            'fgL30': _safe(fg30), 'fgL10': _safe(fg10),
            'fga30': _safe(fga30), 'fga10': _safe(fga10),
            'fg3L30': None, 'fg3L10': None,
            'minL30': _safe(m30), 'minL10': _safe(m10), 'std10': round(std10, 1),
            'dir': direction, 'rawDir': direction, 'conf': round(conf, 3),
            'tier': tier, 'tierLabel': tl, 'units': units, 'avail': 'OK',
            'volume': vol, 'trend': trend, 'fgTrend': _safe(fgTrend), 'minTrend': _safe(minTrend),
            'flags': flags, 'flagsStr': f"{flags}/10", 'flagDetails': flag_details,
            'homeAvgPts': None, 'awayAvgPts': None, 'b2bAvgPts': None,
            'restAvgPts': None, 'b2bDiff': None,
            'recentOver': ro, 'recentUnder': ru,
            'lineSpread': None, 'impliedProb': None, 'edge': None,
            'lineHistory': [{'line': _safe(line), 'batch': 0, 'ts': ''}],
            'predPts':  _safe(round(pred_pts, 1)) if pred_pts else None,
            'predGap':  _safe(round(pred_gap, 1)) if pred_pts else None,
            'preMatchReason': pre_reason,
            'actualPts': _safe(actual), 'result': result, 'delta': _safe(delta),
            'postMatchReason': post_reason, 'lossType': loss_type, 'reason': '',
            'playerModelHR': None, 'playerModelPlays': None,
            'bucketHR': None, 'bucketPlays': None,
            'season': season_label,
        })

    print(f"    Processed {processed}, Analysed {len(plays)}, Skipped {skipped}")
    return plays


def main():
    print("=" * 60)
    print("PropEdge V9.1 — Generate Season JSONs")
    print("=" * 60)
    t0 = time.time()

    # Load all game logs
    df25 = pd.read_csv(FILE_GL_2425, parse_dates=['GAME_DATE'])
    df26 = pd.read_csv(FILE_GL_2526, parse_dates=['GAME_DATE'])
    h2h  = pd.read_csv(FILE_H2H)
    all_logs = pd.concat([df25, df26], ignore_index=True)
    all_logs['GAME_DATE'] = pd.to_datetime(all_logs['GAME_DATE'])

    # Load or train model
    model = None
    if FILE_MODEL.exists():
        with open(FILE_MODEL, 'rb') as f: model = pickle.load(f)
        print(f"  ✓ Loaded model")
    else:
        print("  Training model first (live rolling stats)...")
        from model_trainer import train_and_save
        model = train_and_save(FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_MODEL, FILE_TRUST)

    player_trust = {}
    if FILE_TRUST.exists():
        with open(FILE_TRUST) as f: player_trust = json.load(f)

    # ── 2024-25: Synthetic lines ──
    print(f"\n  Generating 2024-25 synthetic props...")
    synth = generate_season_lines(df25, '2024-25')
    print(f"  Synthetic props: {len(synth)}")

    print(f"  Running model on 2024-25 (live rolling stats)...")
    plays_25 = run_model_on_props(synth, all_logs, h2h, model, player_trust, '2024-25')

    graded_25 = [p for p in plays_25 if p['result'] in ('WIN', 'LOSS')]
    wins_25   = sum(1 for p in graded_25 if p['result'] == 'WIN')
    print(f"  2024-25: {len(plays_25)} plays, {len(graded_25)} graded, "
          f"{wins_25}W/{len(graded_25)-wins_25}L = {wins_25/len(graded_25)*100:.1f}%")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEASON_2425, 'w') as f: json.dump(_clean_for_json(plays_25), f)
    print(f"  ✓ Saved {SEASON_2425.name} ({len(plays_25)} plays) — LOCKED")
    log_event('GEN', 'SEASON_2425_GENERATED', SEASON_2425.name, rows_after=len(plays_25))

    # ── 2025-26: Real prop lines ──
    print(f"\n  Processing 2025-26 real props...")
    props = pd.read_excel(FILE_PROPS, sheet_name='Player_Points_Props', parse_dates=['Date'])
    print(f"  Real props: {len(props)}")

    print(f"  Running model on 2025-26 (live rolling stats)...")
    plays_26 = run_model_on_props(props, all_logs, h2h, model, player_trust, '2025-26')

    graded_26 = [p for p in plays_26 if p['result'] in ('WIN', 'LOSS')]
    wins_26   = sum(1 for p in graded_26 if p['result'] == 'WIN')
    print(f"  2025-26: {len(plays_26)} plays, {len(graded_26)} graded, "
          f"{wins_26}W/{len(graded_26)-wins_26}L = {wins_26/len(graded_26)*100:.1f}%")

    with open(SEASON_2526, 'w') as f: json.dump(_clean_for_json(plays_26), f)
    print(f"  ✓ Saved {SEASON_2526.name} ({len(plays_26)} plays)")
    log_event('GEN', 'SEASON_2526_GENERATED', SEASON_2526.name, rows_after=len(plays_26))

    print(f"\n  Elapsed: {time.time()-t0:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
