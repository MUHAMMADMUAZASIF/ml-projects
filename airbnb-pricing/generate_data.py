"""Generate a synthetic Inside-Airbnb-style listings dataset (10,000 listings).

Prices follow realistic rules (location, room type, size, amenities, reviews)
plus noise, so a model has real signal to learn and SHAP gives sensible answers.
"""
import numpy as np
import pandas as pd

NEIGHBOURHOODS = {  # name: (price multiplier, lat, lon)
    "Downtown": (1.45, 34.050, -118.250), "Beachfront": (1.60, 34.010, -118.495),
    "Hollywood": (1.25, 34.098, -118.327), "Midtown": (1.10, 34.045, -118.340),
    "Arts District": (1.20, 34.040, -118.230), "Suburb North": (0.85, 34.200, -118.450),
    "Suburb East": (0.80, 34.080, -118.100), "Suburb South": (0.78, 33.900, -118.300),
    "Hills": (1.55, 34.120, -118.400), "Old Town": (0.95, 34.060, -118.200),
    "Airport Area": (0.82, 33.945, -118.400), "University": (0.90, 34.070, -118.440),
}
AMENITIES = {"wifi": 0.95, "kitchen": 0.85, "air_conditioning": 0.6, "washer": 0.55,
             "free_parking": 0.45, "gym": 0.15, "pool": 0.12, "hot_tub": 0.08}


def generate(path="data/airbnb_listings.csv", n=10000, seed=42):
    rng = np.random.default_rng(seed)
    names = list(NEIGHBOURHOODS)
    nb = rng.choice(names, n, p=np.array([.12, .05, .10, .10, .07, .09, .08, .08, .05, .10, .06, .10]))
    room = rng.choice(["Entire home/apt", "Private room", "Shared room"], n, p=[.6, .35, .05])
    room_mult = pd.Series(room).map({"Entire home/apt": 1.0, "Private room": 0.55, "Shared room": 0.32}).values

    acc = np.where(room == "Entire home/apt", rng.integers(2, 11, n),
                   np.where(room == "Private room", rng.integers(1, 4, n), rng.integers(1, 3, n)))
    bedrooms = np.clip(np.round(acc / 2.2 + rng.normal(0, .5, n)), 0, 8).astype(int)
    bathrooms = np.clip(np.round(acc / 4 + rng.normal(0, .4, n) * 2) / 2, 0.5, 4)
    am = {k: (rng.random(n) < p).astype(int) for k, p in AMENITIES.items()}
    superhost = (rng.random(n) < .22).astype(int)
    n_reviews = rng.poisson(rng.gamma(2, 15, n))
    score = np.clip(rng.normal(4.7, .25, n) + .1 * superhost, 3.0, 5.0).round(2)
    score = np.where(n_reviews == 0, np.nan, score)
    min_nights = rng.choice([1, 2, 3, 5, 7, 30], n, p=[.4, .25, .15, .08, .05, .07])
    availability = rng.integers(0, 366, n)

    nb_mult = pd.Series(nb).map({k: v[0] for k, v in NEIGHBOURHOODS.items()}).values
    lat = pd.Series(nb).map({k: v[1] for k, v in NEIGHBOURHOODS.items()}).values + rng.normal(0, .01, n)
    lon = pd.Series(nb).map({k: v[2] for k, v in NEIGHBOURHOODS.items()}).values + rng.normal(0, .01, n)

    log_price = (np.log(60) + np.log(nb_mult) + np.log(room_mult)
                 + 0.085 * acc + 0.14 * bedrooms + 0.08 * bathrooms
                 + 0.30 * am["pool"] + 0.18 * am["hot_tub"] + 0.10 * am["air_conditioning"]
                 + 0.07 * am["gym"] + 0.05 * am["free_parking"] + 0.04 * am["kitchen"]
                 + 0.03 * am["washer"] + 0.06 * superhost
                 + 0.35 * (np.nan_to_num(score, nan=4.5) - 4.5)
                 - 0.10 * (min_nights >= 30) + 0.0002 * np.minimum(n_reviews, 100)
                 + rng.normal(0, 0.20, n))
    price = np.round(np.exp(log_price), 0)

    df = pd.DataFrame({"neighbourhood": nb, "room_type": room, "accommodates": acc,
                       "bedrooms": bedrooms, "bathrooms": bathrooms, **am,
                       "host_is_superhost": superhost, "number_of_reviews": n_reviews,
                       "review_score": score, "minimum_nights": min_nights,
                       "availability_365": availability, "latitude": lat.round(5),
                       "longitude": lon.round(5), "price": price})
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = generate()
    print("saved data/airbnb_listings.csv", d.shape, "| median price:", d.price.median())
