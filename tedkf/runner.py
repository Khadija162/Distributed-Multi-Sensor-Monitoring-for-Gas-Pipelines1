"""Batch TeDKF evaluation with per-sensor trust decisions and RMSE evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .model import TeDKF

def evaluate(source="data/public/sensor_measurements.csv",target="artifacts",
             lam=.35,gamma=5):
    df=pd.read_csv(source)
    sensors=sorted(df.sensor_id.unique())
    times=sorted(df.time.unique())
    counts=df.groupby("time").sensor_id.nunique()
    if not (counts==len(sensors)).all():
        raise ValueError("Incomplete sensor snapshots")
    machine=TeDKF(len(sensors),lam=lam,gamma=gamma)
    events=[]
    estimates=[]
    for time in times:
        frame=df[df.time==time].set_index("sensor_id").loc[sensors]
        values=frame[[f"z{i}" for i in range(1,5)]].to_numpy(dtype=float)
        result=machine.step(values)
        truth=frame.iloc[0][[f"truth{i}" for i in range(1,5)]].to_numpy(dtype=float)
        estimates.append(np.mean((np.asarray(result.state)-truth)**2))
        for j,sensor in enumerate(sensors):
            events.append({"time":int(time),"sensor_id":sensor,
                           "fault_label":str(frame.iloc[j]["fault"]),
                           "trusted":bool(result.trusted[j]),
                           "detection_score":float(result.detection_scores[j]),
                           "threshold":result.threshold,
                           "state_error_rmse":float(np.sqrt(estimates[-1]))})
    out=Path(target)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(events).to_parquet(out/"gold"/"diagnostics.parquet",index=False)
    summary={"rmse":float(np.sqrt(np.mean(estimates))),
             "readings":len(events),"time_steps":len(times),
             "flagged":int(sum(not e["trusted"] for e in events)),
             "lambda":lam,"gamma":gamma,
             "note":"Synthetic thermal-model sensor faults; NOT gas-pipeline field validation"}
    (out/"evaluation.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary

if __name__=="__main__":
    print(evaluate())
