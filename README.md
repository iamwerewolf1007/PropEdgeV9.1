# PropEdge V9.1

NBA player prop prediction engine — dual-model, H2H-enhanced, position-weighted.

## What's New in V9.1.6 (Rolling Stats Fix)

### Critical Bug Fixed: Stale Rolling Stats
All rolling averages (L3–L200) are now computed **live** from career-chronological game
history at prediction time. Previously, pre-computed CSV snapshot values were read, which
systematically excluded the most recent game from every player's rolling stats.

**Example — Ryan Rollins (Mar 28 prediction):**
- Old: L3 = 18.0 (pre-game snapshot, excluded his 36-pt Mar 25 game)
- Fixed: L3 = 25.0 (live computation including Mar 25)
- Error eliminated: 7pt per player across every prediction

### New Files
- `rolling_engine.py` — live rolling stat computation (all windows, career-chronological)
- `reasoning_engine.py` — unique pre/post-match reasoning (no templates)

### Architecture Changes
- `batch_predict.py` — uses `rolling_engine.extract_prediction_features()` live
- `generate_season_json.py` — uses live rolling stats for season JSON generation
- `model_trainer.py` — training data uses same live computation as prediction
- `batch0_grade.py` — post-match reasoning + daily Excel grading + cross-check validation

## Daily Architecture

```
BATCH 0 (6AM UK)     → Grade yesterday → append game logs → rebuild H2H → retrain model
                     → cross-check rolling stats vs yesterday's predictions
BATCH 1 (8AM UK)     → Early lines → live rolling stats → daily Excel → today.json
BATCH 2 (6PM UK)     → Main prediction batch
BATCH 3 (1hr pre-tip)→ Final update
```

## File Structure

```
PropEdgeV9.1/
├── daily/                    ← One Excel per game day (YYYY-MM-DD.xlsx)
│   └── Sheet 1: Props (all fresh rolling stats)
│   └── Sheet 2: Predictions (direction, tier, conf, reasoning)
│   └── Sheet 3: Graded (filled by Batch 0)
├── master/                   ← Cumulative post-match CSVs
├── source-files/             ← Raw data (game logs, H2H, prop lines)
├── data/                     ← Dashboard data (today.json, season JSONs)
├── models/                   ← projection_model.pkl, player_trust.json
├── logs/                     ← Batch execution logs
├── rolling_engine.py         ← Live rolling stat computation ← NEW
├── reasoning_engine.py       ← Unique reasoning generator ← NEW
├── config.py                 ← Constants, paths, DVP rankings
├── batch_predict.py          ← Batch 1/2/3 prediction ← FIXED
├── batch0_grade.py           ← Batch 0 grading + cross-check ← FIXED
├── generate_season_json.py   ← Season JSON generation ← FIXED
├── model_trainer.py          ← GBR projection model ← FIXED
├── h2h_builder.py            ← H2H database builder (unchanged)
├── synthetic_lines.py        ← 2024-25 synthetic lines (unchanged)
├── audit.py                  ← Append-only audit log (unchanged)
├── run.py                    ← Orchestrator
├── index.html                ← Dashboard
└── requirements.txt
```

## Execution

```bash
pip install -r requirements.txt

# Initial setup
python3 run.py setup

# Full regeneration (retrain model + rebuild both season JSONs)
python3 run.py generate

# Live prediction test
python3 batch_predict.py 2 2026-03-28
```

## Non-Negotiable Constraints

1. Zero data deletion — append-only
2. Graded plays are immutable (WIN/LOSS/DNP never modified)
3. Cross-season rolling stats — computed career-chronologically, never reset at season boundaries
4. No `groupby().apply()` for rolling stats
5. Always datetime sort, never string sort
6. `PLAYER_NAME` always in column reorder lists
7. numpy int64/float64 converted before json.dump via `_clean_json()`
8. Git push via SSH with `BatchMode=yes`
