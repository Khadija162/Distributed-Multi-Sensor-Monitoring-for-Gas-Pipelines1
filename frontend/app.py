import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Distributed Sensor Monitoring",layout="wide")
st.title("Distributed Multi-Sensor Monitoring")
st.caption("Trust-enhanced distributed Kalman filtering · Innovation-only fusion · Sensor reliability")
st.warning("Research demo based on a thermal system, not a validated natural-gas pipeline monitor.")
api=os.getenv("API_URL","http://localhost:8000")
try:
    response=requests.get(api+"/api/summary",timeout=10)
    response.raise_for_status()
    report=response.json()
except requests.RequestException as exc:
    st.error(f"Backend unavailable: {exc}")
    st.stop()
quality=report.get("quality")
result=report.get("evaluation")
if quality and result:
    a,b,c=st.columns(3)
    a.metric("Sensor nodes",quality["sensors"])
    b.metric("Flagged observations",result["flagged"])
    c.metric("Model RMSE",f"{result['rmse']:.4f}")
    st.info(result["note"])
    try:
        r=requests.get(api+"/api/sensors",timeout=10);r.raise_for_status()
        sensors=pd.DataFrame(r.json()["sensors"])
        st.subheader("Sensor status")
        st.dataframe(sensors,use_container_width=True)
        r=requests.get(api+"/api/diagnostics",params={"limit":1000},timeout=10)
        r.raise_for_status()
        events=pd.DataFrame(r.json()["events"])
        if not events.empty:
            st.subheader("Trust by sensor")
            chart=events.groupby("sensor_id").trusted.mean().sort_values()
            st.bar_chart(chart)
            st.subheader("Recent diagnostic events")
            st.dataframe(events.tail(50),use_container_width=True)
    except requests.RequestException as exc:st.error(str(exc))
else:
    st.info("Run the pipeline and TeDKF evaluation to prepare metrics.")
