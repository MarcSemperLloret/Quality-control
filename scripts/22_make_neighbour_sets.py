"""Neighbour sets for sensitivity: K values and distance rings, over the same quality-filtered pool."""
import os
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
RESULTS = os.environ.get('RESULTS_DIR', str(ROOT / 'results_v3'))
RAW = os.environ.get('RAW_PARQUET', str(ROOT / 'avamet_all.parquet'))
import pandas as pd,numpy as np,json,sys
from pathlib import Path
S=Path(sys.argv[1])
co=pd.read_csv(S/'pool/pool_coords.csv',index_col=0);sample=sorted(pd.read_csv(RESULTS + "/selected_stations.csv").station_id)
D=pd.DataFrame(0.,index=sample,columns=co.index)
for s in sample:
    p1,p2=np.radians(co.loc[s,'lat']),np.radians(co.lat.values);dl=np.radians(co.lon.values-co.loc[s,'lon'])
    D.loc[s]=2*6371.0088*np.arcsin(np.sqrt(np.sin((p2-p1)/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2))
al=pd.read_parquet(S/'pool/pool_all.parquet')
cfg={f'K{k}':(0,1e9,k) for k in (3,7,10)}
cfg.update({'R0_10':(0,10,5),'R10_25':(10,25,5),'R25_50':(25,50,5),'R50_100':(50,100,5)})
for n,(a,b,k) in cfg.items():
    ne={};cnt=[];dd=[]
    for s in sample:
        d=D.loc[s].drop(s);d=d[(d>=a)&(d<b)].sort_values().iloc[:k];ne[s]=list(d.index);cnt.append(len(d));dd+=list(d.values)
    o=S/f'pool_{n}';o.mkdir(exist_ok=True);(o/'neighbors.json').write_text(json.dumps(ne))
    need=sorted({x for v in ne.values() for x in v}-set(sample))
    al[al.station_id.isin(need)].to_parquet(o/'pool_extra.parquet')
    print(n,'neigh/target median',int(np.median(cnt)),'targets with <2',sum(c<2 for c in cnt),'median km',round(float(np.median(dd)),1),'extra',len(need),flush=True)
