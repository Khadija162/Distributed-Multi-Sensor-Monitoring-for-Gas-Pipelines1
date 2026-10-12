"""Local Bronze/Silver/Gold data engineering with quality quarantine."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

MEAS=[f"z{i}" for i in range(1,5)]
TRUTH=[f"truth{i}" for i in range(1,5)]

def run(source="data/public/sensor_measurements.csv",target="artifacts"):
    path=Path(source)
    df=pd.read_csv(path)
    required={"time","sensor_id","fault",*MEAS,*TRUTH}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required-set(df.columns))}")
    out=Path(target)
    for folder in ["bronze","silver","gold"]:(out/folder).mkdir(parents=True,exist_ok=True)
    df.to_parquet(out/"bronze"/"observations.parquet",index=False)
    values=df[MEAS+TRUTH].apply(pd.to_numeric,errors="coerce")
    ok=np.isfinite(values.to_numpy()).all(axis=1)
    valid=pd.Series(ok,index=df.index)&df.time.notna()&df.sensor_id.notna()
    valid&=~df.duplicated(["time","sensor_id"],keep="first")
    clean=df.loc[valid].sort_values(["time","sensor_id"]).copy()
    clean.to_parquet(out/"silver"/"observations.parquet",index=False)
    df.loc[~valid].to_parquet(out/"silver"/"rejected.parquet",index=False)
    stats=clean.groupby("sensor_id",as_index=False).agg(
        readings=("time","count"),fault_labels=("fault",lambda s:int((s!="none").sum())))
    stats.to_parquet(out/"gold"/"sensor_summary.parquet",index=False)
    report={"observations":int(len(df)),"valid":int(len(clean)),
            "rejected":int((~valid).sum()),"sensors":int(clean.sensor_id.nunique()),
            "steps":int(clean.time.nunique()),
            "source":"Original CC0 simulated thermal state, not natural-gas hydraulics"}
    (out/"quality.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

if __name__=="__main__":
    print(run())
