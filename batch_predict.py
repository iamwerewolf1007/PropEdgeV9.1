#!/usr/bin/env python3
"""
PropEdge V9.1 — BATCH 1/2/3: PREDICT
Usage: python3 batch_predict.py [1|2|3] [YYYY-MM-DD]
"""
import pandas as pd
import numpy as np
import json,sys,time,pickle,requests
from pathlib import Path
from datetime import datetime,timedelta

sys.path.insert(0,str(Path(__file__).parent.resolve()))
from config import *
from audit import log_event,log_file_state,verify_no_deletion,log_batch_summary

def _clean_json(obj):
    import numpy as _np
    if isinstance(obj,dict): return {k:_clean_json(v) for k,v in obj.items()}
    if isinstance(obj,list): return [_clean_json(v) for v in obj]
    if isinstance(obj,(_np.integer,)): return int(obj)
    if isinstance(obj,(_np.floating,)): return None if _np.isnan(obj) else round(float(obj),4)
    if isinstance(obj,_np.bool_): return bool(obj)
    if isinstance(obj,float) and obj!=obj: return None  # NaN check
    return obj

BATCH=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1] in ('1','2','3') else 2

def _cc(h,l=''):
    r=h.get('x-requests-remaining','?')
    print(f"    Credits: {r} remaining {l}")
    if r!='?' and int(r)<=CREDIT_ALERT: print(f"    ⚠ LOW CREDITS")

def fetch_props(date_str):
    print(f"\n  Fetching props: {date_str} (Batch {BATCH})...")
    _d=datetime.strptime(date_str,'%Y-%m-%d')
    fr=(_d-timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')
    to=(_d+timedelta(hours=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
    r1=requests.get(f"{ODDS_API_BASE}/sports/{SPORT}/events",
        params={'apiKey':ODDS_API_KEY,'dateFormat':'iso','commenceTimeFrom':fr,'commenceTimeTo':to},timeout=30)
    r1.raise_for_status(); _cc(r1.headers,'events')
    events=[e for e in r1.json()
            if datetime.fromisoformat(e['commence_time'].replace('Z','+00:00')).astimezone(ET).strftime('%Y-%m-%d')==date_str]
    print(f"    {len(events)} games")
    if not events: return {},[]

    games={}; spreads=[]
    for e in events:
        eid=e['id']; hr=e['home_team']; ar=e['away_team']; ts=e['commence_time']
        try: gt=datetime.fromisoformat(ts.replace('Z','+00:00')).astimezone(ET).strftime('%-I:%M %p')+' ET'
        except: gt=''
        ht=resolve_abr(hr);at=resolve_abr(ar)
        games[eid]={'home':ht,'away':at,'home_raw':hr,'away_raw':ar,'gt':gt,'ts':ts,
                    'spread':None,'total':None,'props':{}}
        try:
            r2=requests.get(f"{ODDS_API_BASE}/sports/{SPORT}/events/{eid}/odds",
                params={'apiKey':ODDS_API_KEY,'regions':'us','markets':'player_points,spreads,totals',
                        'oddsFormat':'american','dateFormat':'iso'},timeout=30)
            r2.raise_for_status(); _cc(r2.headers); d=r2.json(); g=games[eid]
            for bm in d.get('bookmakers',[]):
                for m in bm.get('markets',[]):
                    mk=m.get('key','')
                    if mk=='spreads' and g['spread'] is None:
                        for o in m.get('outcomes',[]): 
                            if o.get('name')==hr: g['spread']=o.get('point')
                    elif mk=='totals' and g['total'] is None:
                        for o in m.get('outcomes',[]):
                            if o.get('name','').upper()=='OVER': g['total']=o.get('point')
                    elif mk=='player_points':
                        for o in m.get('outcomes',[]):
                            pl=(o.get('description') or '').strip() or o.get('name','').strip()
                            pt=o.get('point');sd=o.get('name','').upper();pr=o.get('price')
                            if not pl or pt is None: continue
                            if pl not in g['props']: g['props'][pl]={'line':pt,'over':None,'under':None,'books':0}
                            if sd=='OVER': g['props'][pl]['over']=pr;g['props'][pl]['books']+=1
                            elif sd=='UNDER': g['props'][pl]['under']=pr
            if g['spread'] is not None:
                spreads.append({'Date':date_str,'Game':f"{at} @ {ht}",'Home':ht,'Away':at,
                    'Spread (Home)':g['spread'],'Total':g['total'],'Commence':ts,'Book':'Odds API'})
            print(f"    ✓ {at} @ {ht}: {len(g['props'])} props")
            time.sleep(0.3)
        except Exception as ex: print(f"    ✗ {ar} @ {hr}: {ex}"); time.sleep(1)

    tp=sum(len(g['props']) for g in games.values())
    print(f"  Total: {tp} props, {len(games)} games")
    log_event(f'B{BATCH}','PROPS_FETCHED',detail=f'{tp} props, {len(games)} games')
    return games,spreads

def run_predictions(games,date_str):
    print(f"\n  Running predictions...")
    df25=pd.read_csv(FILE_GL_2425,parse_dates=['GAME_DATE'])
    df26=pd.read_csv(FILE_GL_2526,parse_dates=['GAME_DATE'])
    h2h=pd.read_csv(FILE_H2H)
    combined=pd.concat([df25,df26],ignore_index=True)
    combined['GAME_DATE']=pd.to_datetime(combined['GAME_DATE'])
    combined=combined.sort_values(['PLAYER_NAME','GAME_DATE']).reset_index(drop=True)

    model=None
    if FILE_MODEL.exists():
        with open(FILE_MODEL,'rb') as f: model=pickle.load(f)
    pt={}
    if FILE_TRUST.exists():
        with open(FILE_TRUST) as f: pt=json.load(f)

    h2h_lkp={(r['PLAYER_NAME'],r['OPPONENT']):r for _,r in h2h.iterrows()}
    pidx={pn:g.sort_values('GAME_DATE') for pn,g in combined.groupby('PLAYER_NAME')}
    tfga=combined.groupby('OPPONENT')['FGA'].mean()
    pace={t:i+1 for i,(t,_) in enumerate(tfga.sort_values(ascending=False).items())}

    # B2B
    b2b={}
    for pn,g in combined.sort_values('GAME_DATE').groupby('PLAYER_NAME'):
        ds=g['GAME_DATE'].values
        for i in range(len(ds)):
            k=(pn,pd.Timestamp(ds[i]).strftime('%Y-%m-%d'))
            b2b[k]=int((ds[i]-ds[i-1]).astype('timedelta64[D]').astype(int)) if i>0 else 99

    # Existing plays for line movement
    existing=[]
    if TODAY_JSON.exists():
        with open(TODAY_JSON) as f: existing=json.load(f)
    elp={(p['player'],p.get('match','')):(i,p) for i,p in enumerate(existing) if p['date']==date_str}

    batch_ts=now_uk().strftime('%H:%M')
    plays=[];skip_reasons={'low_line':0,'no_player':0,'few_games':0,'no_L30':0,'few_recent':0,'no_play':0}

    for eid,g in games.items():
        ht=g['home'];at=g['away'];ms=f"{at} @ {ht}"
        fms=f"{TEAM_FULL.get(at,at)} @ {TEAM_FULL.get(ht,ht)}"
        sv=g['spread'];tv=g['total'];blow=abs(sv)>=10 if sv else False

        for pname,pd_ in g['props'].items():
            line=pd_.get('line')
            if not line or line<3: skip_reasons['low_line']+=1;continue
            if pname not in pidx: skip_reasons['no_player']+=1;continue
            ph=pidx[pname]; prior=ph[ph['GAME_DATE']<date_str]
            if len(prior)<5: skip_reasons['few_games']+=1;continue
            sn=prior.iloc[-1]
            L30=sn.get('L30_PTS')
            if pd.isna(L30): skip_reasons['no_L30']+=1;continue
            L20=sn.get('L20_PTS',L30);L10=sn.get('L10_PTS',L30);L5=sn.get('L5_PTS',L30);L3=sn.get('L3_PTS',L30)
            for a in [L20,L10,L5,L3]:
                if pd.isna(a): a=L30

            ta=sn.get('GAME_TEAM_ABBREVIATION','');ih=ta==ht;opp=at if ih else ht
            pos=POS_MAP.get(sn.get('PLAYER_POSITION',''),'Forward')
            rp=prior.tail(20)['PTS'].values
            if len(rp)<5: skip_reasons['few_recent']+=1;continue
            r10=rp[-10:];r20=list(rp[-20:].astype(int))
            r20h=list(prior.tail(20)['IS_HOME'].values.astype(int))

            vol=round(float(L30-line),1);trend=round(float(L5-L30),1)
            s10=float(np.std(r10));hr10=round(sum(1 for r in r10 if r>line)/len(r10)*100)
            hr30=round(sum(1 for r in rp if r>line)/len(rp)*100)
            fg30=sn.get('L30_FG_PCT');fg10=sn.get('L10_FG_PCT')
            m30=sn.get('L30_MIN_NUM');m10=sn.get('L10_MIN_NUM')
            if pd.notna(fg30) and fg30<1.5: fg30*=100
            if pd.notna(fg10) and fg10<1.5: fg10*=100
            fgt=round(fg10-fg30,1) if pd.notna(fg10) and pd.notna(fg30) else None
            mnt=round(m10-m30,1) if pd.notna(m10) and pd.notna(m30) else None

            dP=get_dvp(opp,pos);dO=get_def_overall(opp);op=pace.get(opp,15)
            hr_=h2h_lkp.get((pname,opp))
            hG=int(hr_['H2H_GAMES']) if hr_ is not None else 0
            hA=float(hr_['H2H_AVG_PTS']) if hr_ is not None else None
            hTS=float(hr_['H2H_TS_VS_OVERALL']) if hr_ is not None and pd.notna(hr_.get('H2H_TS_VS_OVERALL')) else 0
            hFA=float(hr_['H2H_FGA_VS_OVERALL']) if hr_ is not None and pd.notna(hr_.get('H2H_FGA_VS_OVERALL')) else 0
            hMN=float(hr_['H2H_MIN_VS_OVERALL']) if hr_ is not None and pd.notna(hr_.get('H2H_MIN_VS_OVERALL')) else 0
            hCF=float(hr_['H2H_CONFIDENCE']) if hr_ is not None and pd.notna(hr_.get('H2H_CONFIDENCE')) else 0
            hStr=f"{hA:.1f} ({hG}g)" if hG>=3 and hA else ""
            uh=hG>=3 and hA is not None

            rest=b2b.get((pname,date_str),99);ib2b=1 if rest==1 else 0
            mp=prior.tail(10)['MIN_NUM'];mcv=float(mp.std()/mp.mean()) if mp.mean()>0 else 1
            ppm=float((prior.tail(10)['PTS']/prior.tail(10)['MIN_NUM'].replace(0,np.nan)).mean())
            rmt=float(prior.tail(3)['MIN_NUM'].mean()-mp.mean())
            fpm=float((prior.tail(10)['FGA']/prior.tail(10)['MIN_NUM'].replace(0,np.nan)).mean())

            # Projection
            pp=None;pg=0
            if model:
                from model_trainer import FEATURES
                fd={'l30':L30,'l10':L10,'l5':L5,'l3':L3,'volume':vol,'trend':trend,'std10':s10,
                    'defP':dP,'pace_rank':op,'h2h_ts_dev':hTS,'h2h_fga_dev':hFA,'h2h_min_dev':hMN,
                    'h2h_conf':hCF,'min_cv':mcv,'pts_per_min':ppm if pd.notna(ppm) else 0,
                    'recent_min_trend':rmt if pd.notna(rmt) else 0,'fga_per_min':fpm if pd.notna(fpm) else 0,
                    'is_b2b':ib2b,'rest_days':rest,'consistency':1/(s10+1),'line':line}
                Xp=pd.DataFrame([fd])[FEATURES].fillna(0)
                pp=float(model.predict(Xp)[0]);pg=abs(pp-line)

            # 10-signal composite
            W=POS_WEIGHTS.get(pos,POS_WEIGHTS['Forward'])
            S={1:np.clip((L30-line)/5,-1,1),2:(hr30/100-0.5)*2,3:(hr10/100-0.5)*2,
               4:np.clip((L5-L30)/5,-1,1),5:np.clip(vol/5,-1,1),6:np.clip((dP-15)/15,-1,1),
               7:np.clip((hA-line)/5,-1,1) if uh else 0.0,8:np.clip((15-op)/15,-1,1),
               9:np.clip((fgt or 0)/10,-1,1),10:np.clip((mnt or 0)/5,-1,1)}
            if not uh: tw=sum(w for k,w in W.items() if k!=7);ws=sum(W[k]*S[k] for k in S if k!=7)
            else: tw=sum(W.values());ws=sum(W[k]*S[k] for k in S)
            comp=ws/tw

            dr='OVER' if (pp and pp>line+0.3) or (not pp and comp>0.05) else 'UNDER' if (pp and pp<line-0.3) or (not pp and comp<-0.05) else 'NO PLAY'
            if dr=='NO PLAY': skip_reasons['no_play']+=1;continue

            sc=float(np.clip(0.5+abs(comp)*0.3,0.50,0.85))
            if s10>8: sc-=0.03
            sc=float(np.clip(sc,0.45,0.85))
            pc=float(np.clip(0.5+pg*0.04,0.45,0.90)) if pp else sc
            conf=0.4*sc+0.6*pc

            io=dr!='UNDER'
            fl=0;fds=[]
            for nm,ag,dt in [
                ('Volume',(io and vol>0) or (not io and vol<0),f"{vol:+.1f}"),
                ('HR L30',(io and hr30>50) or (not io and hr30<50),f"{hr30}%"),
                ('HR L10',(io and hr10>50) or (not io and hr10<50),f"{hr10}%"),
                ('Trend',(io and trend>0) or (not io and trend<0),f"{trend:+.1f}"),
                ('Context',(io and vol>-1) or (not io and vol<1),f"vol={vol:+.1f}"),
                ('Defense',(io and dP>15) or (not io and dP<15),f"#{dP}"),
                ('H2H',uh and ((io and hA>line) or (not io and hA<line)),f"{hA:.1f}" if uh else "N/A"),
                ('Pace',(io and op<15) or (not io and op>15),f"#{op}"),
                ('FG Trend',fgt is not None and ((io and fgt>0) or (not io and fgt<0)),f"{fgt:+.1f}%" if fgt else "N/A"),
                ('Min Trend',mnt is not None and ((io and mnt>0) or (not io and mnt<0)),f"{mnt:+.1f}" if mnt else "N/A"),
            ]:
                fl+=1 if ag else 0;fds.append({'name':nm,'agrees':bool(ag),'detail':dt})

            ha=True
            if hTS!=0:
                if dr=='OVER' and hTS<-3: ha=False
                elif dr=='UNDER' and hTS>3: ha=False

            if conf>=0.70 and fl>=8 and s10<=6 and ha: tier=1;tl='T1_ULTRA'
            elif conf>=0.65 and fl>=7 and s10<=7 and ha: tier=1;tl='T1_PREMIUM'
            elif conf>=0.62 and fl>=7 and s10<=7 and ha: tier=1;tl='T1'
            elif conf>=0.55 and fl>=6 and s10<=8 and ha: tier=2;tl='T2'
            else: tier=3;tl='T3'

            tr=pt.get(pname)
            if tr is not None and isinstance(tr,(int,float)) and tr<0.42 and tier==1: tier=2;tl='T2'
            units=3.0 if tl=='T1_ULTRA' else 2.0 if tier==1 else 1.0 if tier==2 else 0.0

            oo=american_to_decimal(pd_.get('over'));uo=american_to_decimal(pd_.get('under'))
            ro=sum(1 for r in r20 if r>line);ru=sum(1 for r in r20 if r<=line)

            # Line history tracking
            lh=[{'line':line,'batch':BATCH,'ts':batch_ts}]
            ekey=(pname,ms)
            if ekey in elp:
                _,ep=elp[ekey]
                old_lh=ep.get('lineHistory',[])
                if isinstance(old_lh,list) and len(old_lh)>0:
                    lh=old_lh
                    if not any(h.get('batch')==BATCH for h in lh if isinstance(h,dict)):
                        lh.append({'line':line,'batch':BATCH,'ts':batch_ts})

            reason=f"Model: {dr} {line} at {int(conf*100)}% ({fl}/10 flags)."
            if pp: reason+=f" Projection: {pp:.1f} pts (gap: {pg:.1f})."
            if hStr: reason+=f" H2H: {hStr}."

            play={
                'date':date_str,'player':pname,'match':ms,'fullMatch':fms,
                'isHome':ih,'team':str(ta),'gameTime':g['gt'],'position':pos,'posSimple':pos[:1],
                'line':line,'overOdds':oo,'underOdds':uo,
                'books':pd_.get('books',1),'minLine':None,'maxLine':None,
                'spread':sv,'total':tv,'blowout':blow,
                'l30':round(float(L30),1),'l20':round(float(L20),1),'l10':round(float(L10),1),
                'l5':round(float(L5),1),'l3':round(float(L3),1),
                'hr30':hr30,'hr10':hr10,
                'recent':r20[:5],'recent10':r20[:10],'recent20':r20,'recent20homes':[bool(x) for x in r20h],
                'defO':dO,'defP':dP,'pace':op,'h2h':hStr,'h2hG':hG,
                'h2hTsDev':hTS,'h2hFgaDev':hFA,'h2hConfidence':hCF,
                'h2hProfile':hr_.get('H2H_SCORING_PROFILE','') if hr_ is not None else '',
                'fgL30':round(fg30,1) if pd.notna(fg30) else None,
                'fgL10':round(fg10,1) if pd.notna(fg10) else None,
                'fga30':round(float(sn.get('L30_FGA',0)),1) if pd.notna(sn.get('L30_FGA')) else None,
                'fga10':round(float(sn.get('L10_FGA',0)),1) if pd.notna(sn.get('L10_FGA')) else None,
                'fg3L30':None,'fg3L10':None,
                'minL30':round(float(m30),1) if pd.notna(m30) else None,
                'minL10':round(float(m10),1) if pd.notna(m10) else None,
                'std10':round(s10,1),'dir':dr,'rawDir':dr,'conf':round(conf,3),
                'tier':tier,'tierLabel':tl,'units':units,'avail':'OK',
                'volume':vol,'trend':trend,'fgTrend':fgt,'minTrend':mnt,
                'flags':fl,'flagsStr':f"{fl}/10",'flagDetails':fds,
                'homeAvgPts':None,'awayAvgPts':None,'b2bAvgPts':None,'restAvgPts':None,'b2bDiff':None,
                'recentOver':ro,'recentUnder':ru,'lineSpread':None,'impliedProb':None,'edge':None,
                'lineHistory':lh,'predPts':round(pp,1) if pp else None,'predGap':round(pg,1) if pp else None,
                'preMatchReason':reason,'actualPts':None,'result':None,'delta':None,'reason':'',
                'playerModelHR':None,'playerModelPlays':None,'bucketHR':None,'bucketPlays':None,
                'season':'2025-26',
            }
            plays.append(play)

    total_skipped = sum(skip_reasons.values())
    print(f"  {len(plays)} predictions (skipped {total_skipped})")
    if total_skipped > 0:
        parts = [f"{v} {k}" for k,v in skip_reasons.items() if v > 0]
        print(f"    Skip breakdown: {', '.join(parts)}")
    log_event(f'B{BATCH}','PREDICTIONS',detail=f'{len(plays)} plays, skipped {total_skipped}: {skip_reasons}')
    return plays

def save_today(plays,date_str):
    """
    Merge new plays into today.json.
    Rules:
    - Graded plays (WIN/LOSS/DNP) are IMMUTABLE — never touched
    - Ungraded plays are UPDATED with latest line/prediction, preserving lineHistory
    - New players not seen before are ADDED
    - Players from previous batch but NOT in current API fetch are KEPT (not dropped)
    - Dedup key: (player, match, date)
    """
    existing=[]
    if TODAY_JSON.exists():
        with open(TODAY_JSON) as f: existing=json.load(f)
    before=len(existing)

    # Separate today's plays from historical
    today_existing = [p for p in existing if p['date']==date_str]
    historical = [p for p in existing if p['date']!=date_str]

    # Build lookup of today's existing plays
    existing_map = {}  # (player, match) → play dict
    for p in today_existing:
        key = (p['player'], p.get('match',''))
        existing_map[key] = p

    # Build lookup of new plays
    new_map = {}
    for p in plays:
        key = (p['player'], p['match'])
        new_map[key] = p

    # Merge: iterate all known keys (existing + new)
    all_keys = set(existing_map.keys()) | set(new_map.keys())
    merged_today = []
    added = 0
    updated = 0
    preserved = 0

    for key in all_keys:
        old = existing_map.get(key)
        new = new_map.get(key)

        if old and old.get('result') in ('WIN','LOSS','DNP'):
            # GRADED — immutable, keep as-is
            merged_today.append(old)
            continue

        if old and new:
            # EXISTS + NEW DATA → update, preserve lineHistory
            old_lh = old.get('lineHistory', [])
            new_line = new['line']
            # Append to history if line changed or different batch
            if isinstance(old_lh, list) and len(old_lh) > 0:
                new['lineHistory'] = old_lh
                if not any(isinstance(h,dict) and h.get('batch')==BATCH for h in old_lh):
                    new['lineHistory'].append({'line':new_line,'batch':BATCH,'ts':batch_ts})
                else:
                    # Same batch re-run — update the line value for this batch
                    for h in new['lineHistory']:
                        if isinstance(h,dict) and h.get('batch')==BATCH:
                            h['line'] = new_line
                            h['ts'] = batch_ts
            merged_today.append(new)
            updated += 1

        elif old and not new:
            # EXISTS but NOT in current API fetch → KEEP (don't drop)
            merged_today.append(old)
            preserved += 1

        elif new and not old:
            # BRAND NEW player/prop
            merged_today.append(new)
            added += 1

    # Sort: T1 first, then by confidence
    merged_today.sort(key=lambda p:(p.get('tier',9),-p.get('conf',0)))
    all_p = merged_today + sorted(historical, key=lambda p:p['date'], reverse=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(TODAY_JSON,'w') as f: json.dump(_clean_json(all_p),f)

    t1=sum(1 for p in merged_today if p.get('tier')==1)
    t2=sum(1 for p in merged_today if p.get('tier')==2)
    print(f"\n  ✓ today.json: {len(merged_today)} today ({t1} T1, {t2} T2)")
    print(f"    Added: {added} | Updated: {updated} | Preserved: {preserved}")
    log_batch_summary(f'B{BATCH}',props_fetched=len(plays),plays_added=added)
    verify_no_deletion(f'B{BATCH}',TODAY_JSON,before,len(all_p),'SAVE_TODAY')

def main():
    date_str=today_et()
    if len(sys.argv)>2 and '-' in sys.argv[2]: date_str=sys.argv[2]

    print("="*60)
    print(f"PropEdge V9.1 — BATCH {BATCH}: PREDICT")
    print(f"  Date: {date_str} | {now_uk().strftime('%Y-%m-%d %H:%M %Z')}")
    print("="*60)
    log_event(f'B{BATCH}','BATCH_START',detail=date_str)

    games,_=fetch_props(date_str)
    if not games: print("  No games."); return
    plays=run_predictions(games,date_str)
    save_today(plays,date_str)

    repo=REPO_DIR if REPO_DIR.exists() else ROOT
    from batch0_grade import git_push,notify
    git_push(repo,f"B{BATCH}: {date_str} — {len(plays)} plays")
    log_event(f'B{BATCH}','BATCH_COMPLETE')
    notify("PropEdge",f"B{BATCH}: {len(plays)} plays")

if __name__=='__main__': main()
