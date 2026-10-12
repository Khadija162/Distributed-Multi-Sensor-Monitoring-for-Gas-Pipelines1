"""Original CC0 deterministic thermal-network sensor data with injected faults."""
from pathlib import Path
import numpy as np
import pandas as pd
from .model import A

KINDS=("bias","drift","noise","stuck","none")

def generate(output="data/public/sensor_measurements.csv",steps=100,n_sensors=10,
             fault="bias",faulty=4,onset=35,seed=42):
    if fault not in KINDS or n_sensors<3 or not 0<=faulty<=n_sensors:
        raise ValueError("Invalid simulation configuration")
    rng=np.random.default_rng(seed)
    state=np.array([20.,22.,24.,23.])
    rows=[]
    freeze=None
    for time in range(steps):
        state=A@state+rng.normal(0,.05,4)
        for sensor in range(n_sensors):
            observed=state+rng.normal(0,.12,4)
            compromised=sensor<faulty and time>=onset
            if compromised:
                if fault=="bias": observed+=1.8
                elif fault=="drift": observed+=min(2.5,(time-onset)*.09)
                elif fault=="noise": observed+=rng.normal(0,1.8,4)
                elif fault=="stuck":
                    # Stuck at sensor's previous reading from a nominal step.
                    observed=np.array(freeze[sensor],dtype=float)
            rows.append({"time":time,"sensor_id":f"S{sensor+1:02d}",
                         **{f"z{i+1}":float(observed[i]) for i in range(4)},
                         **{f"truth{i+1}":float(state[i]) for i in range(4)},
                         "fault":fault if compromised and fault!="none" else "none"})
        if time==onset-1:
            freeze=[r_[f"z{i+1}"] for r_ in rows[-n_sensors:] for i in range(4)]
            freeze=np.asarray(freeze).reshape(n_sensors,4)
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)
    return path

if __name__=="__main__":
    print(generate())
