"""
PropEdge V9.1 — Reasoning Engine
==================================
Generates unique, data-driven pre-match and post-match reasoning.
No generic templates. Every reasoning driven by actual signals.

Pre-match:
1. Lead with strongest evidence (single most compelling driver)
2. Name specific matchup/statistical context
3. Reference agreeing AND disagreeing signals by name
4. Flag specific risk factor

Post-match:
1. Outcome with margin
2. Causal factor
3. Which model signals were right/wrong
4. Loss classification
"""
import numpy as np
import pandas as pd


# ─── Loss Classification Labels ──────────────────────────────
LOSS_TYPES = {
    'CLOSE_CALL': 'within 2 pts of line',
    'MINUTES_SHORTFALL': 'played fewer minutes than expected',
    'SHOOTING_VARIANCE': 'normal minutes but hot/cold shooting',
    'BLOWOUT_EFFECT': 'game out of hand, starters pulled',
    'MODEL_CORRECT': 'prediction direction was sound',
    'MODEL_FAILURE': 'signals were wrong for this player/matchup',
}


def _describe_dvp(rank):
    """Describe defensive rank in plain English."""
    if rank <= 5: return f"elite defense (#{rank})"
    if rank <= 10: return f"strong defense (#{rank})"
    if rank <= 15: return f"average defense (#{rank})"
    if rank <= 22: return f"below-average defense (#{rank})"
    return f"weak defense (#{rank})"


def _describe_pace(rank):
    if rank <= 5: return f"fast pace (#{rank})"
    if rank <= 12: return f"above-average pace (#{rank})"
    if rank <= 20: return f"moderate pace (#{rank})"
    return f"slow pace (#{rank})"


def _describe_h2h_profile(profile_str, h2h_ts_dev, h2h_avg, player_name, opp, n_games):
    """Generate H2H context sentence."""
    if not h2h_avg or n_games < 3:
        return None
    dev_str = f"{abs(h2h_ts_dev):.1f}%"
    direction = "higher" if h2h_ts_dev > 0 else "lower"
    
    if abs(h2h_ts_dev) >= 5:
        return (f"His TS% runs {dev_str} {direction} than overall in {n_games} meetings vs "
                f"{opp} — a meaningful efficiency shift in this matchup.")
    elif abs(h2h_ts_dev) >= 2:
        return (f"Mild TS% deviation ({'+' if h2h_ts_dev > 0 else ''}{h2h_ts_dev:.1f}%) in "
                f"{n_games} meetings vs {opp}.")
    return (f"TS% is consistent with overall averages in {n_games} meetings vs {opp} "
            f"(no significant efficiency shift).")


def _strongest_signal(flag_details, direction, vol, trend, h2h_avg, line, std10, L30):
    """Identify the single most compelling signal to lead with."""
    is_over = 'UNDER' not in direction
    
    # Score each signal for lead candidacy
    candidates = []
    
    # Volume (L30 vs line gap)
    vol_abs = abs(vol)
    if vol_abs >= 3:
        agrees = (is_over and vol > 0) or (not is_over and vol < 0)
        candidates.append((vol_abs * (1.5 if agrees else 1.2), 'volume',
                          f"L30 average of {L30:.1f} pts is {vol_abs:.1f} {'above' if vol > 0 else 'below'} the {line} line"))
    
    # H2H if meaningful
    if h2h_avg is not None:
        h2h_gap = abs(h2h_avg - line)
        if h2h_gap >= 2:
            agrees = (is_over and h2h_avg > line) or (not is_over and h2h_avg < line)
            candidates.append((h2h_gap * (1.4 if agrees else 1.1), 'h2h',
                              f"H2H avg of {h2h_avg:.1f} pts vs this opponent ({'+' if h2h_avg > line else ''}{h2h_avg - line:.1f} vs line)"))
    
    # Trend
    trend_abs = abs(trend)
    if trend_abs >= 2:
        agrees = (is_over and trend > 0) or (not is_over and trend < 0)
        candidates.append((trend_abs * (1.3 if agrees else 1.0), 'trend',
                          f"recent scoring {'up' if trend > 0 else 'down'} {trend_abs:.1f} pts vs L30 (L5 vs L30 trend)"))
    
    # Consistency (std10 low = strong signal)
    if std10 <= 5:
        candidates.append((6 - std10, 'consistency',
                          f"low scoring variance (σ={std10:.1f}) — highly predictable output"))
    
    if not candidates:
        return f"L30 average of {L30:.1f} pts vs line of {line}"
    
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][2]


def generate_pre_match_reason(play_data):
    """
    Generate unique, data-driven pre-match reasoning.
    
    play_data must contain: player, dir, line, l30, l10, l5, l3,
    vol, trend, flags, flagDetails, h2h, h2hG, h2hTsDev, h2hFgaDev,
    h2hProfile, defP, defO, pace, fgTrend, minTrend, std10,
    minL10, minL30, conf, predPts, predGap, tierLabel, position,
    match, recent, hr30, hr10
    """
    p = play_data
    direction = p.get('dir', 'OVER')
    is_over = 'UNDER' not in direction
    is_lean = 'LEAN' in direction
    line = p.get('line', 0)
    player = p.get('player', '')
    opp = p.get('match', '').split(' @ ')[-1] if ' @ ' in p.get('match', '') else ''
    if p.get('isHome') and ' @ ' in p.get('match', ''):
        parts = p.get('match', '').split(' @ ')
        opp = parts[0]
    
    L30 = p.get('l30', 0)
    L10 = p.get('l10', L30)
    L5  = p.get('l5', L30)
    L3  = p.get('l3', L30)
    vol = p.get('volume', L30 - line)
    trend = p.get('trend', L5 - L30)
    std10 = p.get('std10', 5.0)
    flags = p.get('flags', 0)
    flag_details = p.get('flagDetails', [])
    
    h2hG = p.get('h2hG', 0)
    h2h_str = p.get('h2h', '')
    h2h_ts_dev = p.get('h2hTsDev', 0)
    h2h_avg = None
    if h2hG >= 3 and h2h_str:
        try: h2h_avg = float(h2h_str.split(' ')[0])
        except: pass
    
    defP = p.get('defP', 15)
    pace = p.get('pace', 15)
    fgTrend = p.get('fgTrend')
    minTrend = p.get('minTrend')
    minL30 = p.get('minL30')
    minL10 = p.get('minL10')
    predPts = p.get('predPts')
    predGap = p.get('predGap')
    hr30 = p.get('hr30', 50)
    hr10 = p.get('hr10', 50)
    recent = p.get('recent', [])
    conf = p.get('conf', 0.55)
    
    # Identify agreeing vs disagreeing signals
    agrees = [f for f in flag_details if f.get('agrees')]
    disagrees = [f for f in flag_details if not f.get('agrees')]
    agree_names = [f['name'] for f in agrees]
    disagree_names = [f['name'] for f in disagrees]
    
    # ── Lead sentence: strongest evidence ──
    lead = _strongest_signal(flag_details, direction, vol, trend, h2h_avg, line, std10, L30)
    
    parts = []
    last_name = player.split()[-1] if player else player
    
    # S1: Lead with strongest signal
    dir_word = "supports" if (
        (is_over and vol > 0) or (not is_over and vol < 0)
    ) else "works against"
    
    if 'L30 average' in lead and 'h2h' not in lead.lower():
        parts.append(
            f"{last_name}'s {lead}, {dir_word} the {direction.replace('LEAN ', 'lean ')}."
        )
    elif 'H2H' in lead or 'h2h' in lead.lower():
        dir_text = 'supporting' if ((is_over and h2h_avg and h2h_avg > line) or (not is_over and h2h_avg and h2h_avg < line)) else 'working against'
        parts.append(
            f"{last_name} has a {lead} — {dir_text} the {direction.replace('LEAN ', 'lean ')}."
        )
    else:
        parts.append(f"{last_name} shows {lead}.")
    
    # S2: Matchup/efficiency context
    matchup_parts = []
    if h2hG >= 3 and h2h_avg is not None and abs(h2h_avg - L30) >= 1.5:
        diff = h2h_avg - L30
        matchup_parts.append(
            f"His H2H avg of {h2h_avg:.1f} pts is {abs(diff):.1f} "
            f"{'above' if diff > 0 else 'below'} his season L30"
        )
    if abs(h2h_ts_dev) >= 3 and h2hG >= 3:
        matchup_parts.append(
            f"his TS% shifts {'+' if h2h_ts_dev > 0 else ''}{h2h_ts_dev:.1f}% in this matchup"
        )
    if fgTrend is not None and abs(fgTrend) >= 3:
        matchup_parts.append(
            f"FG% trend {'+' if fgTrend > 0 else ''}{fgTrend:.1f}% (L10 vs L30)"
        )
    
    def_desc = _describe_dvp(defP)
    pace_desc = _describe_pace(pace)
    context_str = f"Opponent ranks as {def_desc} with {pace_desc}."
    if matchup_parts:
        parts.append(f"{'; '.join(s[0].upper() + s[1:] for s in matchup_parts[:2])}. {context_str}")
    else:
        parts.append(context_str)
    
    # S3: Signal consensus with named signals
    if len(agrees) >= 7:
        top_agree = ', '.join(agree_names[:4])
        if disagree_names:
            top_disagree = ', '.join(disagree_names[:2])
            parts.append(
                f"{flags}/10 signals agree ({top_agree} among the strongest); "
                f"{top_disagree} {'dissent' if len(disagrees) > 1 else 'dissents'}."
            )
        else:
            parts.append(f"Full consensus: {flags}/10 signals align ({top_agree} all agree).")
    elif len(agrees) >= 5:
        top_agree = ', '.join(agree_names[:3])
        top_disagree = ', '.join(disagree_names[:2]) if disagree_names else 'none'
        parts.append(
            f"{flags}/10 signals agree — {top_agree} support {direction.split(' ')[-1].lower()}; "
            f"{top_disagree + ' against' if disagree_names else 'no significant counter-signals'}."
        )
    else:
        parts.append(
            f"Mixed signal picture: {flags}/10 agree. "
            f"{'Agreeing: ' + ', '.join(agree_names[:3]) + '.' if agree_names else ''} "
            f"{'Counter-signals: ' + ', '.join(disagree_names[:3]) + '.' if disagree_names else ''}"
        )
    
    # S4: Model prediction context
    if predPts is not None:
        gap_str = f"{predGap:.1f} pts {'above' if predPts > line else 'below'} line"
        parts.append(
            f"Projection model targets {predPts:.1f} pts ({gap_str}; "
            f"{int(conf*100)}% blended confidence)."
        )
    
    # S5: Risk factor — what could make this wrong
    risks = []
    
    # Biggest risk: recent L3 surge if UNDER, or slump if OVER
    l3_vs_l30 = L3 - L30
    if is_over and l3_vs_l30 < -4:
        risks.append(
            f"L3 has dropped to {L3:.1f} ({abs(l3_vs_l30):.1f} below L30) — "
            f"if the slump deepens, this over is vulnerable"
        )
    elif not is_over and l3_vs_l30 > 4:
        last_score = recent[0] if recent else None
        score_str = f" after a {last_score}-pt performance" if last_score else ""
        risks.append(
            f"L3 has surged to {L3:.1f}{score_str} ({l3_vs_l30:.1f} above L30) — "
            f"if that momentum carries, this under is at risk"
        )
    
    # Variance risk
    if std10 > 7:
        risks.append(f"high scoring volatility (σ={std10:.1f}) introduces outcome uncertainty")
    
    # Hits rate risk
    if is_over and hr30 < 45:
        risks.append(f"only {hr30}% hit rate over L30 — line may be set fairly")
    elif not is_over and hr30 > 55:
        risks.append(f"{hr30}% hit rate over L30 suggests line may already price in OVER tendency")
    
    # Minutes risk
    if minL10 is not None and minL30 is not None:
        min_diff = minL10 - minL30
        if min_diff < -3:
            risks.append(
                f"minutes trending down ({minL10:.1f} L10 vs {minL30:.1f} L30 — "
                f"role reduction would suppress counting stats"
            )
    
    if risks:
        parts.append(f"Risk: {risks[0]}.")
    
    # Add LEAN qualifier at start if lean
    result = ' '.join(p for p in parts if p.strip())
    if is_lean:
        result = f"[Low conviction — lean only] " + result
    
    return result


def generate_post_match_reason(play_data):
    """
    Generate post-match reasoning after actual result is known.
    
    Requires: actualPts, result, delta, dir, line, l30, l10, l5, l3,
    minL10, predPts, flags, flagDetails, h2hTsDev, player
    
    Returns: (reason_str, loss_type_str)
    """
    p = play_data
    direction = p.get('dir', 'OVER')
    is_over = 'UNDER' not in direction
    result = p.get('result', '')
    actual = p.get('actualPts')
    line = p.get('line', 0)
    delta = p.get('delta', 0) or 0
    player = p.get('player', '')
    last_name = player.split()[-1] if player else player
    
    if actual is None or result not in ('WIN', 'LOSS'):
        return '', None
    
    margin = abs(delta)
    direction_word = direction.replace('LEAN ', '').lower()
    result_word = 'WIN' if result == 'WIN' else 'LOSS'
    
    # S1: Outcome with margin
    outcome_parts = [
        f"{result_word}: {direction_word.upper()} {'hit' if result == 'WIN' else 'missed'} "
        f"({actual} pts vs {line} line, {'+' if delta > 0 else ''}{delta:.1f})."
    ]
    
    # S2: Classify and explain cause
    loss_type = None
    
    if result == 'WIN':
        # WIN — identify why it worked
        L30 = p.get('l30', 0)
        L3 = p.get('l3', L30)
        predPts = p.get('predPts')
        
        if predPts is not None and abs(actual - predPts) <= 3:
            outcome_parts.append(
                f"Projection model called {predPts:.1f} pts — actual {actual} was {abs(actual - predPts):.1f} off. "
                f"Model accuracy strong on this play."
            )
        
        flag_details = p.get('flagDetails', [])
        agrees = [f['name'] for f in flag_details if f.get('agrees')]
        if len(agrees) >= 7:
            outcome_parts.append(f"Signal alignment was sound — {len(agrees)}/10 signals were correct.")
        
        loss_type = 'MODEL_CORRECT'
        
    else:
        # LOSS — diagnose cause
        L30 = p.get('l30', 0)
        predPts = p.get('predPts')
        flag_details = p.get('flagDetails', [])
        disagrees = [f['name'] for f in flag_details if not f.get('agrees')]
        agrees = [f['name'] for f in flag_details if f.get('agrees')]
        
        # Close call?
        if margin <= 2:
            loss_type = 'CLOSE_CALL'
            outcome_parts.append(
                f"Missed by {margin:.1f} pts — a close call. "
                f"Direction was {'sound, narrowly missed execution' if len(agrees) >= 6 else 'questionable (signals were mixed)'}."
            )
        
        # Blowout (large delta, wrong way, low actual for OVER or high for UNDER)
        elif is_over and actual < line - 8:
            loss_type = 'BLOWOUT_EFFECT'
            outcome_parts.append(
                f"Underperformed by {margin:.1f} pts — potentially a blowout or garbage-time scenario "
                f"that pulled starters early."
            )
        elif not is_over and actual > line + 8:
            loss_type = 'BLOWOUT_EFFECT'
            outcome_parts.append(
                f"Overperformed by {margin:.1f} pts — unexpected scoring burst, possibly vs "
                f"a blowout or pace mismatch."
            )
        
        # Projection model was on right track but missed
        elif predPts is not None:
            pred_gap = abs(predPts - line)
            pred_correct_direction = (is_over and predPts > line) or (not is_over and predPts < line)
            if pred_correct_direction and pred_gap >= 1:
                loss_type = 'SHOOTING_VARIANCE'
                outcome_parts.append(
                    f"Projection model targeted {predPts:.1f} pts (correct direction) but actual "
                    f"was {actual}. Likely shooting variance — per-minute output may have been on pace."
                )
            else:
                loss_type = 'MODEL_FAILURE'
                outcome_parts.append(
                    f"Projection model ({predPts:.1f} pts) and actual ({actual}) both missed the mark. "
                    f"Key signals that disagreed: {', '.join(disagrees[:3]) if disagrees else 'none noted'}."
                )
        else:
            # No projection model — judge by signal quality
            if len(agrees) >= 7 and margin > 3:
                loss_type = 'SHOOTING_VARIANCE'
                outcome_parts.append(
                    f"Signals were strongly aligned ({len(agrees)}/10) but result went against them. "
                    f"Variance loss — statistical regression expected."
                )
            elif len(disagrees) >= 5:
                loss_type = 'MODEL_FAILURE'
                outcome_parts.append(
                    f"Counter-signals ({', '.join(disagrees[:3])}) proved accurate. "
                    f"Model read this play incorrectly."
                )
            else:
                loss_type = 'CLOSE_CALL' if margin <= 3 else 'SHOOTING_VARIANCE'
                outcome_parts.append(
                    f"Mixed-signal play went wrong by {margin:.1f} pts. "
                    f"Classification: marginal loss."
                )
        
        # S3: Which signals were right vs wrong
        if len(disagrees) > 0:
            outcome_parts.append(
                f"Counter-signals that proved correct: {', '.join(disagrees[:3])}."
            )
    
    reason = ' '.join(s for s in outcome_parts if s.strip())
    return reason, loss_type


def classify_loss_type(play_data):
    """Standalone loss classifier for batch grading. Returns loss_type string."""
    _, lt = generate_post_match_reason(play_data)
    return lt
