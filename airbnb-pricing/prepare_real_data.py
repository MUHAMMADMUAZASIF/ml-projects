"""Convert a REAL Inside Airbnb `listings.csv` (detailed version, listings.csv.gz) into the
schema used by train.py.  Download from http://insideairbnb.com/get-the-data/
Usage: python prepare_real_data.py listings.csv.gz data/airbnb_listings.csv
NOTE: column names in Inside Airbnb change slightly between releases - adjust if needed.
"""
import sys

import numpy as np
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
d = pd.read_csv(src)
d["price"] = d["price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
am = d["amenities"].fillna("").str.lower()
out = pd.DataFrame({
    "neighbourhood": d["neighbourhood_cleansed"], "room_type": d["room_type"],
    "accommodates": d["accommodates"], "bedrooms": d["bedrooms"].fillna(1),
    "bathrooms": d.get("bathrooms", pd.Series(1, index=d.index)).fillna(1),
    "wifi": am.str.contains("wifi").astype(int), "kitchen": am.str.contains("kitchen").astype(int),
    "air_conditioning": am.str.contains("air conditioning").astype(int),
    "washer": am.str.contains("washer").astype(int),
    "free_parking": am.str.contains("free parking|free street parking").astype(int),
    "gym": am.str.contains("gym").astype(int), "pool": am.str.contains("pool").astype(int),
    "hot_tub": am.str.contains("hot tub").astype(int),
    "host_is_superhost": (d["host_is_superhost"] == "t").astype(int),
    "number_of_reviews": d["number_of_reviews"], "review_score": d["review_scores_rating"],
    "minimum_nights": d["minimum_nights"], "availability_365": d["availability_365"],
    "latitude": d["latitude"], "longitude": d["longitude"], "price": d["price"],
})
out = out[(out.price > 10) & (out.price < out.price.quantile(0.99))].dropna(subset=["price"])
out.to_csv(dst, index=False)
print("saved", dst, out.shape)
