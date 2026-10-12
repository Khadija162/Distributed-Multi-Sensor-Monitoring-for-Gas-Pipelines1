import numpy as np
from fastapi.testclient import TestClient
from backend.main import app
from tedkf.model import A,TeDKF,mutual_reachability,trusted_innovations
from tedkf.simulation import generate
from tedkf.pipeline import run
from tedkf.runner import evaluate

def test_state_transition():
    assert A.shape==(4,4)
    assert abs(A[0,0]-.8)<1e-8

def test_idb():
    reference=np.array([[0,0,0,0],[.01,0,0,0],[.02,0,0,0],[.03,0,0,0],[.04,0,0,0],[.05,0,0,0]])
    scores=mutual_reachability(reference,np.zeros(4),gamma=5)
    assert scores.min()>=0
    trusted,_,threshold=trusted_innovations(np.vstack([reference[:2],np.ones(4)*10]),reference,lam=.1)
    assert trusted[-1]==False and trusted[:2].all()
    assert threshold>0

def test_filter_finite():
    filt=TeDKF()
    for _ in range(12):
        event=filt.step(np.ones((10,4)))
        assert np.isfinite(event.state).all()
        assert len(event.trusted)==10

def test_pipeline(tmp_path):
    source=generate(tmp_path/"sensor.csv",steps=50,faulty=7)
    report=run(source,tmp_path/"output")
    outcome=evaluate(source,tmp_path/"output")
    assert report["sensors"]==10
    assert outcome["time_steps"]==50
    assert np.isfinite(outcome["rmse"])

def test_api_missing(tmp_path,monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR",str(tmp_path))
    client=TestClient(app)
    assert client.get("/health").status_code==200
    assert client.get("/api/diagnostics").json()["events"]==[]
    assert client.get("/api/diagnostics?limit=0").status_code==422
