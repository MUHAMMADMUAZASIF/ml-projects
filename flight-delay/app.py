"""Streamlit page where anyone can check if their flight will be late.
Run:  streamlit run app.py      (after `python train.py`)
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")
bundle = joblib.load("outputs/model.joblib")
model, meta = bundle["model"], bundle["meta"]
cats = meta["categories"]

st.title("✈️ Will my flight be late?")
c1, c2, c3 = st.columns(3)
airline = c1.selectbox("Airline", cats["Reporting_Airline"])
origin = c2.selectbox("From", cats["Origin"])
dest = c3.selectbox("To", [d for d in cats["Dest"] if d != origin])
c1, c2, c3 = st.columns(3)
month = c1.selectbox("Month", list(range(1, 13)), index=5)
dow = c2.selectbox("Day of week (1=Mon)", list(range(1, 8)))
hour = c3.slider("Scheduled departure hour", 0, 23, 17)

weather = {}
if meta["weather_cols"]:
    st.write("**Weather at the departure airport**")
    w1, w2, w3, w4 = st.columns(4)
    weather = {"precip_mm": w1.number_input("Rain (mm)", 0.0, 100.0, 0.0),
               "wind_kts": w2.number_input("Wind (kts)", 0, 80, 8),
               "visibility_mi": w3.number_input("Visibility (mi)", 0.25, 10.0, 10.0),
               "temp_f": w4.number_input("Temp (F)", -20, 120, 65)}
    weather = {k: v for k, v in weather.items() if k in meta["weather_cols"]}

if st.button("Check my flight", type="primary"):
    dist = meta["route_distance"].get(f"{origin}-{dest}", meta["mean_distance"])
    row = {"Reporting_Airline": airline, "Origin": origin, "Dest": dest, "Month": month,
           "DayOfWeek": dow, "dep_hour": hour, "Distance": dist, **weather}
    X = pd.DataFrame([row])
    for c in cats:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    X = X[model.feature_names_in_] if hasattr(model, "feature_names_in_") else X
    p = float(model.predict_proba(X)[0, 1])
    st.metric("Chance of arriving 15+ min late", f"{p:.0%}")
    st.progress(min(p, 1.0))
    if p >= meta["best_threshold"]:
        st.error("Likely delayed - consider leaving buffer time for connections.")
    else:
        st.success("Likely on time.")
