"""
PropEdge V9.1 — Projection Model Trainer
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


def build_training_data(file_2425, file_2526, file_h2h):
    df25 = pd.read_csv(file_2425, parse_dates=['GAME_DATE'])
    df26 = pd.read_csv(file_2526, parse_dates=['GAME_DATE'])
    h2h = pd.read_csv(file_h2h)
    combined = pd.concat([df25, df26], ignore_index=True)
    combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'])
    combined = combined.sort_values(['PLAYER_NAME','GAME_DATE']).reset_index(drop=True)
    h2h_lkp = {(r['PLAYER_NAME'], r['OPPONENT']): r for _, r in h2h.iterrows()}

    rows = []
    for pname, grp in combined.groupby('PLAYER_NAME'):
        grp = grp.sort_values('GAME_DATE').reset_index(drop=True)
        for i in range(10, len(grp)):
            row = grp.iloc[i]
            p = grp.iloc[max(0,i-30):i]; p10 = grp.iloc[max(0,i-10):i]
            p5 = grp.iloc[max(0,i-5):i]; p3 = grp.iloc[max(0,i-3):i]
            l30=p['PTS'].mean(); l10=p10['PTS'].mean(); l5=p5['PTS'].mean(); l3=p3['PTS'].mean()
            std10 = p10['PTS'].std() if len(p10)>=3 else 5.0
            line = round(l30*2)/2
            opp = row['OPPONENT']; pos = POS_MAP.get(row.get('PLAYER_POSITION',''),'Forward')
            hr = h2h_lkp.get((pname,opp))
            h2h_ts = float(hr['H2H_TS_VS_OVERALL']) if hr is not None and pd.notna(hr.get('H2H_TS_VS_OVERALL')) else 0
            h2h_fga = float(hr['H2H_FGA_VS_OVERALL']) if hr is not None and pd.notna(hr.get('H2H_FGA_VS_OVERALL')) else 0
            h2h_min = float(hr['H2H_MIN_VS_OVERALL']) if hr is not None and pd.notna(hr.get('H2H_MIN_VS_OVERALL')) else 0
            h2h_conf = float(hr['H2H_CONFIDENCE']) if hr is not None and pd.notna(hr.get('H2H_CONFIDENCE')) else 0
            mp = p10['MIN_NUM']
            min_cv = mp.std()/mp.mean() if mp.mean()>0 else 1
            ppm = (p10['PTS']/p10['MIN_NUM'].replace(0,np.nan)).mean()
            rmt = p3['MIN_NUM'].mean()-p10['MIN_NUM'].mean()
            fpm = (p10['FGA']/p10['MIN_NUM'].replace(0,np.nan)).mean()
            rest = (grp.iloc[i]['GAME_DATE']-grp.iloc[i-1]['GAME_DATE']).days if i>0 else 99
            rows.append({
                'player':pname,'date':row['GAME_DATE'],
                'l30':round(l30,1),'l10':round(l10,1),'l5':round(l5,1),'l3':round(l3,1),
                'volume':round(l30-line,1),'trend':round(l5-l30,1),'std10':round(std10,1),
                'defP':get_dvp(opp,pos),'pace_rank':15,
                'h2h_ts_dev':h2h_ts,'h2h_fga_dev':h2h_fga,'h2h_min_dev':h2h_min,'h2h_conf':h2h_conf,
                'min_cv':round(min_cv,3),'pts_per_min':round(ppm,2) if pd.notna(ppm) else 0,
                'recent_min_trend':round(rmt,1) if pd.notna(rmt) else 0,
                'fga_per_min':round(fpm,2) if pd.notna(fpm) else 0,
                'is_b2b':1 if rest==1 else 0,'rest_days':rest,
                'consistency':round(1/(std10+1),3),'line':line,'actual_pts':row['PTS'],
            })
    return pd.DataFrame(rows)


def train_and_save(file_2425, file_2526, file_h2h, model_file, trust_file):
    print("    Building training data...")
    train_df = build_training_data(file_2425, file_2526, file_h2h)
    print(f"    Training samples: {len(train_df)}")
    X = train_df[FEATURES].fillna(0); y = train_df['actual_pts']
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       min_samples_leaf=20, random_state=42)
    model.fit(X, y)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with open(model_file, 'wb') as f: pickle.dump(model, f)
    print(f"    ✓ Model → {model_file.name}")

    train_df['pred'] = model.predict(X)
    train_df['correct'] = (
        ((train_df['pred']>train_df['line'])&(train_df['actual_pts']>train_df['line'])) |
        ((train_df['pred']<train_df['line'])&(train_df['actual_pts']<train_df['line']))
    ).astype(int)
    trust = {p: round(g['correct'].mean(),3) for p,g in train_df.groupby('player') if len(g)>=10}
    with open(trust_file,'w') as f: json.dump(trust, f, indent=2)
    print(f"    ✓ Trust: {len(trust)} players → {trust_file.name}")
    return model
