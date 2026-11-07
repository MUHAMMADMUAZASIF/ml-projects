"""Streamlit app: tells a host what a listing should cost and what it is 'leaving on the table'.
Run:  streamlit run app.py      (after `python train.py`)
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Airbnb Pricing Engine", page_icon="🏠")
bundle = joblib.load("outputs/model.joblib")
model, meta = bundle["model"], bundle["meta"]


def predict(row):
    X = pd.DataFrame([row])
    X = pd.get_dummies(X, columns=["neighbourhood", "room_type"]).reindex(columns=meta["columns"], fill_value=0).astype(float)
    return float(__import__("numpy").expm1(model.predict(X)[0]))


st.title("🏠 Airbnb Pricing Engine")
c1, c2 = st.columns(2)
nb = c1.selectbox("Neighbourhood", meta["neighbourhoods"])
room = c2.selectbox("Room type", meta["room_types"])
acc = c1.slider("Guests it sleeps", 1, 12, 4)
beds = c2.slider("Bedrooms", 0, 8, 2)
baths = c1.slider("Bathrooms", 0.5, 4.0, 1.0, 0.5)
superhost = c2.checkbox("Superhost")
reviews = c1.number_input("Number of reviews", 0, 1000, 20)
score = c2.slider("Review score", 3.0, 5.0, 4.7, 0.05)
min_n = c1.selectbox("Minimum nights", [1, 2, 3, 5, 7, 30])
avail = c2.slider("Availability (days/year)", 0, 365, 200)
st.write("**Amenities**")
cols = st.columns(4)
am = {a: int(cols[i % 4].checkbox(a.replace("_", " ").title(), value=a in ("wifi", "kitchen")))
      for i, a in enumerate(meta["amenities"])}
current = st.number_input("Your current nightly price ($)", 10, 5000, 100)

ll = meta["nb_latlon"][nb]
row = {"neighbourhood": nb, "room_type": room, "accommodates": acc, "bedrooms": beds, "bathrooms": baths,
       **am, "host_is_superhost": int(superhost), "number_of_reviews": reviews, "review_score": score,
       "minimum_nights": min_n, "availability_365": avail, "latitude": ll["latitude"], "longitude": ll["longitude"]}

if st.button("Predict price", type="primary"):
    p = predict(row)
    st.metric("Suggested nightly price", f"${p:,.0f}", f"{p - current:+,.0f} vs your price")
    if p > current * 1.05:
        st.warning(f"You may be leaving about ${p - current:,.0f} per night on the table.")
    elif p < current * 0.95:
        st.info("You are priced above the model's estimate - expect fewer bookings.")
    else:
        st.success("Your price is in line with similar listings.")
    # what-if: value of each amenity you don't have yet
    rows = []
    for a in meta["amenities"]:
        if not am[a]:
            r2 = {**row, a: 1}
            rows.append({"Add this amenity": a.replace("_", " ").title(), "Extra $/night": round(predict(r2) - p, 2)})
    if rows:
        st.write("**What adding an amenity could add to your price**")
        st.dataframe(pd.DataFrame(rows).sort_values("Extra $/night", ascending=False), hide_index=True)
