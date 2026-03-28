# PropEdge V9.1

NBA Player Points Prop Prediction Engine.

## Quick Start

```bash
cd ~/Documents/GitHub/PropEdgeV9.1
pip install -r requirements.txt

# 1. One-time: generate season JSONs + train model
python3 run.py generate

# 2. Daily: grade + predict
python3 run.py 0      # Grade last night
python3 run.py 2      # Main prediction batch

# 3. Install automated scheduling
python3 run.py setup
```

## Batch Schedule (UK Time)

| Batch | Time | Purpose |
|-------|------|---------|
| 0 | 06:00 | Grade last night, update logs, rebuild H2H, retrain model |
| 1 | 08:00 | Early lines fetch |
| 2 | 18:00 | Main prediction run |
| 3 | Dynamic (1hr pre-tip) | Final update |

## File Structure

```
PropEdgeV9.1/
├── config.py              # Constants, DVP, team maps, paths
├── audit.py               # Data integrity control system
├── run.py                 # Master runner + launchd + git init
├── batch0_grade.py        # BATCH 0: grade + update + retrain
├── batch_predict.py       # BATCH 1/2/3: fetch odds + predict
├── generate_season_json.py # One-time season JSON generation
├── synthetic_lines.py     # Sportsbook-style line generator (2024-25)
├── h2h_builder.py         # H2H database (vectorised)
├── model_trainer.py       # GBR projection model
├── index.html             # Dashboard
├── source-files/          # Data files
├── data/                  # today.json, season JSONs, audit log
├── models/                # projection_model.pkl, player_trust.json
└── logs/                  # Batch execution logs
```

## Data Integrity

- Audit log: data/audit_log.csv tracks every operation
- Zero-deletion guarantee: no rows are ever removed
- Graded plays are immutable (locked after grading)
- Before/after row counts logged for every file change

## First Run Sequence

1. `python3 run.py setup` — initialise git, install launchd
2. `python3 run.py generate` — generates season_2024_25.json, season_2025_26.json, trains model
3. `python3 run.py 2` — run today's predictions (or `run.py 0` first if there are games to grade)
