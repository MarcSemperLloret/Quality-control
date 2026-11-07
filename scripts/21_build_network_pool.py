"""Network neighbour pool: stations with >=50% valid hours for pressure and gust, <=1% hard-invalid TRAIN pressure."""
import os
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
RESULTS = os.environ.get('RESULTS_DIR', str(ROOT / 'results_v3'))
RAW = os.environ.get('RAW_PARQUET', str(ROOT / 'avamet_all.parquet'))
import duckdb,pandas as pd,numpy as np,json,sys
from pathlib import Path
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
raw=RAW
sample=sorted(pd.read_csv(RESULTS + "/selected_stations.csv").station_id)
c=duckdb.connect();c.execute(f"SET temp_directory='{out.as_posix()}/tmp'")
num=lambda k:f"""TRY_CAST(REPLACE(NULLIF(TRIM(json_extract_string(payload_json,'$."{k}"')),'None'),',','.') AS DOUBLE)"""
t="""TRY_CAST(json_extract_string(payload_json,'$."data ini"') AS TIMESTAMP)"""
df=c.execute(f"""WITH r AS (SELECT station_id,{t} t,{num('pres mit_mit')} p,{num('vent vel_mit')} wm,{num('vent vel_max')} wg,row_number() over () i FROM read_parquet('{raw}'))
 SELECT station_id,t timestamp_utc,arg_max(p,CASE WHEN p IS NOT NULL THEN i END) pressure_raw,AVG(wm) wind_mean,MAX(wg) wind_gust FROM r
 WHERE t>=TIMESTAMP '2023-01-01' AND t<TIMESTAMP '2026-01-01' AND station_id IS NOT NULL GROUP BY 1,2""").df()
df['timestamp_utc']=pd.to_datetime(df.timestamp_utc,utc=True)
tr=df[df.timestamp_utc<'2025-01-01']
q=tr.groupby('station_id').pressure_raw.quantile([.05,.5,.95]).unstack();q.columns=['p5','p50','p95']
u=pd.Series('hPa',index=q.index);u[q.p50.between(80000,120000)&(q.p95>90000)]='Pa';u[q.p50.between(600,820)&(q.p95<900)&(q.p5>400)]='mmHg'
un=df.station_id.map(u).fillna('hPa')
df['pressure_hpa']=np.where(un=='Pa',df.pressure_raw/100,np.where(un=='mmHg',df.pressure_raw*1.3332239,df.pressure_raw))
bad=(df.pressure_hpa<=500)|(df.pressure_hpa>=1100)
hard=bad[df.timestamp_utc<'2025-01-01'].groupby(df.station_id).mean()
df.loc[bad,'pressure_hpa']=np.nan
df.loc[(df.wind_gust<0)|(df.wind_mean<0),['wind_gust','wind_mean']]=np.nan
df.loc[df.wind_mean>df.wind_gust,['wind_gust','wind_mean']]=np.nan
df['h']=df.timestamp_utc.dt.floor('h');H=3*365.25*24
hp=df.dropna(subset=['pressure_hpa']).groupby('station_id').h.nunique()/H
hw=df.dropna(subset=['wind_gust']).groupby('station_id').h.nunique()/H
ok=sorted(set(hp[hp>=.5].index)&set(hw[hw>=.5].index)&set(hard[hard<=.01].index))
co=c.execute(f"select station_id,median(lat) lat,median(lon) lon from read_parquet('{raw}') group by 1").df().set_index('station_id').dropna()
pool=sorted((set(ok)&set(co.index))|set(sample))
print('stations',df.station_id.nunique(),'pressure>=50%',int((hp>=.5).sum()),'gust>=50%',int((hw>=.5).sum()),'hard<=1%',int((hard<=.01).sum()),'pool',len(pool))
cc=co.loc[pool];ne={};ds=[]
for s in sample:
    p1,p2=np.radians(cc.loc[s,'lat']),np.radians(cc.lat.values);dl=np.radians(cc.lon.values-cc.loc[s,'lon'])
    d=pd.Series(2*6371.0088*np.arcsin(np.sqrt(np.sin((p2-p1)/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2)),index=cc.index).drop(s).sort_values().iloc[:5]
    ne[s]=list(d.index);ds.append(d.values)
ds=np.array(ds);print('km 1st NN median',round(float(np.median(ds[:,0])),1),'5th median',round(float(np.median(ds[:,4])),1),'5th max',round(float(ds[:,4].max()),1))
(out/'neighbors.json').write_text(json.dumps(ne,indent=0))
co.loc[pool].join(c.execute(f"select station_id,median(alt) alt from read_parquet('{raw}') group by 1").df().set_index('station_id')).to_csv(out/'pool_coords.csv')
df[df.station_id.isin(pool)][['station_id','timestamp_utc','pressure_hpa','wind_gust','wind_mean']].to_parquet(out/'pool_all.parquet')
need=sorted({n for v in ne.values() for n in v}-set(sample));print('extra neighbours',len(need))
df[df.station_id.isin(need)][['station_id','timestamp_utc','pressure_hpa','wind_gust','wind_mean']].to_parquet(out/'pool_extra.parquet')
