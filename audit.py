"""
PropEdge V9.1 — Audit & Control System
========================================
Append-only integrity log. Tracks every data operation.
Zero-deletion guarantee enforcement.
"""
import pandas as pd
import json, os
from datetime import datetime
from pathlib import Path
from config import AUDIT_LOG, DATA_DIR, now_uk


def _ensure_log():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not AUDIT_LOG.exists():
        pd.DataFrame(columns=[
            'timestamp','batch','operation','file','rows_before','rows_after',
            'rows_added','rows_removed','detail','status'
        ]).to_csv(AUDIT_LOG, index=False)


def log_event(batch, operation, file_name='', rows_before=0, rows_after=0,
              rows_added=0, rows_removed=0, detail='', status='OK'):
    """Append a single audit event."""
    _ensure_log()
    entry = pd.DataFrame([{
        'timestamp': now_uk().strftime('%Y-%m-%d %H:%M:%S'),
        'batch': batch,
        'operation': operation,
        'file': file_name,
        'rows_before': rows_before,
        'rows_after': rows_after,
        'rows_added': rows_added,
        'rows_removed': rows_removed,
        'detail': detail,
        'status': status,
    }])
    entry.to_csv(AUDIT_LOG, mode='a', header=False, index=False)


def log_file_state(batch, file_path, label=''):
    """Log current state of a file (row count, size)."""
    p = Path(file_path)
    if not p.exists():
        log_event(batch, f'FILE_CHECK:{label}', p.name, detail='FILE NOT FOUND', status='WARN')
        return 0

    if p.suffix == '.json':
        with open(p) as f:
            data = json.load(f)
        count = len(data) if isinstance(data, list) else 1
    elif p.suffix == '.csv':
        count = sum(1 for _ in open(p)) - 1  # Minus header
    else:
        count = 0

    size_kb = p.stat().st_size / 1024
    log_event(batch, f'FILE_STATE:{label}', p.name, rows_before=count,
              detail=f'{size_kb:.1f}KB')
    return count


def verify_no_deletion(batch, file_path, rows_before, rows_after, operation):
    """Verify no unexpected row deletions occurred."""
    removed = rows_before - rows_after + (rows_after - rows_before)  # Net
    actual_removed = max(0, rows_before - rows_after)

    if actual_removed > 0:
        log_event(batch, operation, Path(file_path).name,
                  rows_before=rows_before, rows_after=rows_after,
                  rows_removed=actual_removed,
                  detail=f'⚠ {actual_removed} ROWS REMOVED — INVESTIGATE',
                  status='ALERT')
        print(f"  ⚠ AUDIT ALERT: {actual_removed} rows removed from {Path(file_path).name}")
        print(f"    Before: {rows_before}, After: {rows_after}")
        return False

    log_event(batch, operation, Path(file_path).name,
              rows_before=rows_before, rows_after=rows_after,
              rows_added=rows_after - rows_before,
              status='OK')
    return True


def log_batch_summary(batch, props_fetched=0, plays_added=0, plays_graded=0,
                       wins=0, losses=0, dnp=0):
    """Log a batch-level summary."""
    detail = (f'fetched={props_fetched}, added={plays_added}, '
              f'graded={plays_graded}, W={wins}, L={losses}, DNP={dnp}')
    log_event(batch, 'BATCH_SUMMARY', detail=detail)


def get_daily_report(date_str=None):
    """Print a human-readable daily control report."""
    if not AUDIT_LOG.exists():
        print("  No audit log found.")
        return

    df = pd.read_csv(AUDIT_LOG)
    if date_str:
        df = df[df['timestamp'].str.startswith(date_str)]

    print(f"\n  === AUDIT REPORT {'(' + date_str + ')' if date_str else ''} ===")
    print(f"  Total events: {len(df)}")

    alerts = df[df['status'] == 'ALERT']
    if len(alerts) > 0:
        print(f"  ⚠ ALERTS: {len(alerts)}")
        for _, a in alerts.iterrows():
            print(f"    {a['timestamp']} | {a['operation']} | {a['detail']}")
    else:
        print(f"  ✓ No alerts")

    summaries = df[df['operation'] == 'BATCH_SUMMARY']
    for _, s in summaries.iterrows():
        print(f"  {s['timestamp']} | Batch {s['batch']} | {s['detail']}")
