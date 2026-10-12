"""Read-only diagnostic API; never used to control any industrial pipeline."""
import json
import os
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
app=FastAPI(title="TeDKF Distributed Sensor Fault Monitoring",version="0.1.0")

def root():return Path(os.getenv("ARTIFACT_DIR","artifacts"))

def read_json(name):
    p=root()/name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

@app.get("/health")
def health():return {"status":"ok","evaluation_ready":read_json("evaluation.json") is not None}

@app.get("/api/summary")
def summary():return {"quality":read_json("quality.json"),"evaluation":read_json("evaluation.json")}

@app.get("/api/sensors")
def sensors():
    p=root()/"gold"/"sensor_summary.parquet"
    return {"sensors":pd.read_parquet(p).to_dict(orient="records") if p.is_file() else []}

@app.get("/api/diagnostics")
def diagnostics(limit:int=100):
    from fastapi import HTTPException
    if not 1<=limit<=1000:raise HTTPException(422,"limit must be 1-1000")
    p=root()/"gold"/"diagnostics.parquet"
    if not p.is_file():return {"events":[]}
    df=pd.read_parquet(p).tail(limit)
    return {"events":df.where(pd.notna(df),None).to_dict(orient="records")}
