#!/usr/bin/env python3
"""
PropEdge V9.1 — BATCH 0: GRADE + UPDATE (6:00 AM UK)
"""
import pandas as pd
import numpy as np
import json, sys, time, re, subprocess
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import *
from audit import log_event, log_file_state, verify_no_deletion, log_batch_summary

def _clean_json(obj):
    if isinstance(obj,dict): return {k:_clean_json(v) for k,v in obj.items()}
    if isinstance(obj,list): return [_clean_json(v) for v in obj]
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return None if np.isnan(obj) else round(float(obj),4)
    if isinstance(obj,np.bool_): return bool(obj)
    if isinstance(obj,float) and obj!=obj: return None
    return obj


def notify(title, msg):
    try: subprocess.run(['osascript','-e',f'display notification "{msg}" with title "{title}"'],capture_output=True,timeout=5)
    except: pass

def git_push(repo, message):
    """Git add, commit, push via SSH. Non-blocking — won't hang on credential prompts."""
    env = {**__import__('os').environ, 'GIT_SSH_COMMAND': 'ssh -o BatchMode=yes -o StrictHostKeyChecking=no'}
    try:
        subprocess.run(['git','add','-A'], cwd=repo, capture_output=True, timeout=30, env=env)
        r2 = subprocess.run(['git','commit','-m',message], cwd=repo, capture_output=True, timeout=30, env=env)
        if r2.returncode != 0:
            msg = r2.stderr.decode().strip() if r2.stderr else ''
            if 'nothing to commit' in msg:
                print(f"  ✓ Git: nothing to commit")
                return
        # Try regular push first, fall back to --set-upstream
        r3 = subprocess.run(['git','push'], cwd=repo, capture_output=True, timeout=60, env=env)
        if r3.returncode != 0:
            err = r3.stderr.decode().strip() if r3.stderr else ''
            if 'no upstream' in err or 'set-upstream' in err:
                r3 = subprocess.run(['git','push','--set-upstream','origin','main'],
                                    cwd=repo, capture_output=True, timeout=60, env=env)
        if r3.returncode == 0:
            print(f"  ✓ Git push: {message}")
        else:
            err = r3.stderr.decode().strip() if r3.stderr else 'unknown error'
            print(f"  ⚠ Git push failed: {err[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Git push timed out (60s) — check SSH setup")
    except Exception as e:
        print(f"  ⚠ Git error: {e}")

def _si(v):
    try: return int(v) if pd.notna(v) else 0
    except: return 0

def _pm(v):
    s=str(v).strip()
    if s in ('','None','nan','0','PT00M00.00S'): return 0.0
    if s.startswith('PT') and 'M' in s:
        m=re.match(r'PT(\d+)M([\d.]+)S',s)
        return float(m.group(1))+float(m.group(2))/60 if m else 0.0
    if ':' in s: p=s.split(':'); return float(p[0])+float(p[1])/60
    try: return float(s)
    except: return 0.0


def fetch_boxscores(date_str):
    from nba_api.stats.endpoints import ScoreboardV3, BoxScoreTraditionalV3
    print(f"\n  Fetching box scores: {date_str}...")
    time.sleep(1)
    sb=ScoreboardV3(game_date=date_str,league_id='00')
    gh=sb.game_header.get_data_frame(); ls=sb.line_score.get_data_frame()
    if gh.empty: print("    No games"); return []
    gids=gh['gameId'].tolist(); print(f"    {len(gids)} games")

    ctx={}
    for g in gids:
        r=ls[ls['gameId']==g]
        if len(r)>=2: ctx[str(g)]={'htid':r.iloc[0]['teamId'],'ht':r.iloc[0]['teamTricode'],
            'at':r.iloc[1]['teamTricode'],'hs':_si(r.iloc[0].get('score',0)),'as':_si(r.iloc[1].get('score',0))}

    df26=pd.read_csv(FILE_GL_2526)
    bio={}; bc=['PLAYER_ID','PLAYER_NAME','PLAYER_POSITION','PLAYER_POSITION_FULL',
        'PLAYER_CURRENT_TEAM','GAME_TEAM_ABBREVIATION','GAME_TEAM_NAME',
        'PLAYER_HEIGHT','PLAYER_WEIGHT','PLAYER_EXPERIENCE','PLAYER_COUNTRY',
        'PLAYER_DRAFT_YEAR','PLAYER_DRAFT_ROUND','PLAYER_DRAFT_NUMBER']
    for _,r in df26.drop_duplicates('PLAYER_ID',keep='last')[bc].iterrows(): bio[r['PLAYER_ID']]=r.to_dict()

    rows=[]
    for g in gids:
        time.sleep(0.8)
        try:
            box=BoxScoreTraditionalV3(game_id=g); ps=box.player_stats.get_data_frame()
            if ps.empty: continue
            vm={'personId':'PLAYER_ID','teamId':'TEAM_ID','teamTricode':'TEAM_ABBREVIATION',
                'firstName':'FN','familyName':'LN','minutes':'MR','fieldGoalsMade':'FGM',
                'fieldGoalsAttempted':'FGA','threePointersMade':'FG3M','threePointersAttempted':'FG3A',
                'freeThrowsMade':'FTM','freeThrowsAttempted':'FTA',
                'reboundsOffensive':'OREB','reboundsDefensive':'DREB','reboundsTotal':'REB',
                'assists':'AST','steals':'STL','blocks':'BLK','turnovers':'TOV',
                'foulsPersonal':'PF','points':'PTS','plusMinusPoints':'PLUS_MINUS'}
            ps=ps.rename(columns={k:v for k,v in vm.items() if k in ps.columns})
            if 'PLAYER_NAME' not in ps.columns and 'FN' in ps.columns:
                ps['PLAYER_NAME']=ps['FN'].fillna('')+' '+ps['LN'].fillna('')
            c=ctx.get(str(g),{})
            for _,p in ps.iterrows():
                mn=_pm(p.get('MR',0))
                if mn<=0: continue
                pid=_si(p.get('PLAYER_ID',0)); tid=_si(p.get('TEAM_ID',0))
                ta=str(p.get('TEAM_ABBREVIATION',''))
                ih=1 if tid==c.get('htid') else 0
                opp=ps[ps['TEAM_ID']!=tid]['TEAM_ABBREVIATION'].iloc[0] if len(ps[ps['TEAM_ID']!=tid])>0 else 'UNK'
                mu=f"{ta} vs. {opp}" if ih else f"{ta} @ {opp}"
                wl=('W' if c.get('hs',0)>c.get('as',0) else 'L') if ih else ('W' if c.get('as',0)>c.get('hs',0) else 'L')
                pts=_si(p.get('PTS',0));fgm=_si(p.get('FGM',0));fga=_si(p.get('FGA',0))
                fg3m=_si(p.get('FG3M',0));fg3a=_si(p.get('FG3A',0));ftm=_si(p.get('FTM',0));fta=_si(p.get('FTA',0))
                oreb=_si(p.get('OREB',0));dreb=_si(p.get('DREB',0));reb=_si(p.get('REB',0))
                ast=_si(p.get('AST',0));stl=_si(p.get('STL',0));blk=_si(p.get('BLK',0))
                tov=_si(p.get('TOV',0));pf=_si(p.get('PF',0));pm=_si(p.get('PLUS_MINUS',0))
                fgp=fgm/fga if fga>0 else 0;f3p=fg3m/fg3a if fg3a>0 else 0;ftp=ftm/fta if fta>0 else 0
                efg=(fgm+0.5*fg3m)/fga if fga>0 else 0;tsa=2*(fga+0.44*fta)
                ts=pts/tsa if tsa>0 else 0;usg=(fga+0.44*fta+tov)/(mn/5) if mn>0 else 0
                pra=pts+reb+ast;ddc=sum(1 for x in [pts,reb,ast,stl,blk] if x>=10)
                dd=1 if ddc>=2 else 0;td=1 if ddc>=3 else 0
                fp=pts+1.25*reb+1.5*ast+2*stl+2*blk-0.5*tov+0.5*fg3m+1.5*dd+3*td
                b=bio.get(pid,{})
                rows.append({'PLAYER_ID':pid,'PLAYER_NAME':p.get('PLAYER_NAME',b.get('PLAYER_NAME','')),
                    'SEASON':'2025-26','SEASON_TYPE':'Regular Season',
                    'PLAYER_POSITION':b.get('PLAYER_POSITION',''),'PLAYER_POSITION_FULL':b.get('PLAYER_POSITION_FULL',''),
                    'PLAYER_CURRENT_TEAM':b.get('PLAYER_CURRENT_TEAM',ta),
                    'GAME_TEAM_ABBREVIATION':ta,'GAME_TEAM_NAME':b.get('GAME_TEAM_NAME',''),
                    'PLAYER_HEIGHT':b.get('PLAYER_HEIGHT',''),'PLAYER_WEIGHT':b.get('PLAYER_WEIGHT',0),
                    'PLAYER_EXPERIENCE':b.get('PLAYER_EXPERIENCE',0),'PLAYER_COUNTRY':b.get('PLAYER_COUNTRY',''),
                    'PLAYER_DRAFT_YEAR':b.get('PLAYER_DRAFT_YEAR',0),'PLAYER_DRAFT_ROUND':b.get('PLAYER_DRAFT_ROUND',0),
                    'PLAYER_DRAFT_NUMBER':b.get('PLAYER_DRAFT_NUMBER',0),
                    'GAME_ID':int(g),'GAME_DATE':date_str,'MATCHUP':mu,'OPPONENT':opp,
                    'IS_HOME':ih,'WL':wl,'WL_WIN':1 if wl=='W' else 0,'WL_LOSS':1 if wl=='L' else 0,
                    'GAMES_PLAYED_SEASON_RUNNING':0,'MIN':int(round(mn)),'MIN_NUM':round(mn,1),
                    'FGM':fgm,'FGA':fga,'FG_PCT':round(fgp,4),'FG3M':fg3m,'FG3A':fg3a,'FG3_PCT':round(f3p,4),
                    'FTM':ftm,'FTA':fta,'FT_PCT':round(ftp,4),'OREB':oreb,'DREB':dreb,'REB':reb,
                    'AST':ast,'STL':stl,'BLK':blk,'TOV':tov,'PF':pf,'PTS':pts,'PLUS_MINUS':pm,
                    'VIDEO_AVAILABLE':1,'EFF_FG_PCT':round(efg,4),'TRUE_SHOOTING_PCT':round(ts,4),
                    'USAGE_APPROX':round(usg,2),'PTS_REB_AST':pra,'PTS_REB':pts+reb,'PTS_AST':pts+ast,
                    'REB_AST':reb+ast,'DOUBLE_DOUBLE':dd,'TRIPLE_DOUBLE':td,'FANTASY_PTS':round(fp,2),
                    'SEASON_ID':22025})
        except Exception as e: print(f"    ✗ {g}: {e}")
    print(f"  Fetched {len(rows)} logs")
    log_event('B0','BOXSCORES_FETCHED',detail=f'{len(rows)} logs for {date_str}')
    return rows


def append_gamelogs(new_rows):
    if not new_rows: return
    df25=pd.read_csv(FILE_GL_2425,parse_dates=['GAME_DATE'])
    df26=pd.read_csv(FILE_GL_2526,parse_dates=['GAME_DATE'])
    rows_before=len(df26)
    log_file_state('B0',FILE_GL_2526,'BEFORE_APPEND')

    ndf=pd.DataFrame(new_rows); ndf['GAME_DATE']=pd.to_datetime(ndf['GAME_DATE'])
    for w in WINDOWS:
        for c in ROLL_COLS: ndf[f'L{w}_{c}']=np.nan
    hist=pd.concat([df25,df26],ignore_index=True); hist['GAME_DATE']=pd.to_datetime(hist['GAME_DATE'])
    ndf=ndf.sort_values(['PLAYER_ID','GAME_DATE']).reset_index(drop=True)

    for pid in ndf['PLAYER_ID'].unique():
        ph=hist[hist['PLAYER_ID']==pid].sort_values('GAME_DATE')
        pn=ndf[ndf['PLAYER_ID']==pid].sort_values('GAME_DATE')
        for i,(idx,r) in enumerate(pn.iterrows()):
            prior=pd.concat([ph,pn.iloc[:i]]).sort_values('GAME_DATE') if i>0 else ph
            for w in WINDOWS:
                wd=prior.tail(w)
                for c in ROLL_COLS:
                    if len(wd)==0 or (len(wd)<w and w>=100): ndf.at[idx,f'L{w}_{c}']=np.nan
                    elif c in wd.columns:
                        v=wd[c].mean(); ndf.at[idx,f'L{w}_{c}']=round(v,4) if pd.notna(v) else np.nan

    for c in df26.columns:
        if c not in ndf.columns: ndf[c]=np.nan
    ndf=ndf[df26.columns]
    upd=pd.concat([df26,ndf],ignore_index=True).sort_values(['PLAYER_NAME','GAME_DATE'])
    b4=len(upd); upd=upd.drop_duplicates(subset=['PLAYER_ID','GAME_ID'],keep='last')
    if len(upd)<b4: print(f"  Dedup: removed {b4-len(upd)}")

    upd.to_csv(FILE_GL_2526,index=False)
    verify_no_deletion('B0',FILE_GL_2526,rows_before,len(upd),'APPEND_GAMELOGS')
    print(f"  ✓ Game logs: {rows_before} → {len(upd)} (+{len(upd)-rows_before})")


def grade_plays(date_str, new_rows):
    """Grade plays in season_2025_26.json and today.json."""
    results_map={(r['PLAYER_NAME'],date_str):r['PTS'] for r in new_rows}
    graded_count=0; wins=0; losses=0; dnp=0

    for fpath in [SEASON_2526, TODAY_JSON]:
        if not fpath.exists(): continue
        with open(fpath) as f: plays=json.load(f)
        changed=False
        for p in plays:
            if p['date']!=date_str: continue
            if p.get('result') in ('WIN','LOSS'): continue  # Already locked
            lkp=results_map.get((p['player'],date_str))
            if lkp is None: p['result']='DNP';p['actualPts']=None;dnp+=1;changed=True;continue
            actual=int(lkp); p['actualPts']=actual; p['delta']=round(actual-p['line'],1)
            d = p.get('dir','')
            if 'OVER' in d: p['result']='WIN' if actual>p['line'] else 'LOSS'
            elif 'UNDER' in d: p['result']='WIN' if actual<p['line'] else 'LOSS'
            else: p['result']='DNP'; dnp+=1; changed=True; continue
            if p['result']=='WIN': wins+=1
            elif p['result']=='LOSS': losses+=1
            graded_count+=1; changed=True
        if changed:
            with open(fpath,'w') as f: json.dump(_clean_json(plays),f)

    print(f"  ✓ Graded {graded_count} plays: {wins}W/{losses}L, {dnp} DNP")
    log_batch_summary('B0',plays_graded=graded_count,wins=wins,losses=losses,dnp=dnp)


def main():
    print("="*60)
    print(f"PropEdge V9.1 — BATCH 0: GRADE + UPDATE")
    print(f"  {now_uk().strftime('%Y-%m-%d %H:%M %Z')}")
    print("="*60)
    log_event('B0','BATCH_START')

    yesterday=(datetime.now(ET)-timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"  Grading: {yesterday}")

    rows=fetch_boxscores(yesterday)
    append_gamelogs(rows)
    grade_plays(yesterday, rows)

    print("  Rebuilding H2H...")
    from h2h_builder import build_h2h
    build_h2h(FILE_GL_2425,FILE_GL_2526,FILE_H2H)

    print("  Retraining model...")
    from model_trainer import train_and_save
    train_and_save(FILE_GL_2425,FILE_GL_2526,FILE_H2H,FILE_MODEL,FILE_TRUST)

    repo=REPO_DIR if REPO_DIR.exists() else ROOT
    git_push(repo,f"B0: grade {yesterday}")
    log_event('B0','BATCH_COMPLETE')
    notify("PropEdge V9.1",f"B0 done — graded {yesterday}")
    print("  ✓ BATCH 0 complete")

if __name__=='__main__': main()
