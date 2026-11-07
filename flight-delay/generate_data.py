"""Generate a synthetic BTS-style US flight dataset (100,000 flights).

Column names follow the real BTS 'On-Time Performance' file (Reporting_Airline, Origin, Dest,
Month, DayOfWeek, CRSDepTime, Distance, ArrDel15) plus origin-airport weather columns,
because BTS itself has no weather (you join it from NOAA / Meteostat for real data).
ArrDel15 = 1 if the flight arrived 15+ minutes late.
"""
import numpy as np
import pandas as pd

AIRLINES = {"WN": .00, "DL": -.35, "AA": .10, "UA": .12, "B6": .30, "NK": .35,
            "AS": -.30, "F9": .32, "OO": .05, "MQ": .25}
AIRPORTS = {"ATL": (33.6, -84.4, .10), "ORD": (41.98, -87.9, .35), "DFW": (32.9, -97.0, .12),
            "DEN": (39.86, -104.7, .15), "LAX": (33.94, -118.4, .10), "JFK": (40.64, -73.78, .40),
            "EWR": (40.69, -74.17, .45), "SFO": (37.62, -122.4, .38), "SEA": (47.45, -122.3, .02),
            "LAS": (36.08, -115.15, .00), "MCO": (28.43, -81.3, .12), "CLT": (35.21, -80.94, .05),
            "PHX": (33.43, -112.0, -.15), "MIA": (25.79, -80.29, .15), "BOS": (42.36, -71.0, .25),
            "MSP": (44.88, -93.2, -.05), "DTW": (42.21, -83.35, -.02), "PHL": (39.87, -75.24, .20),
            "LGA": (40.78, -73.87, .30), "BWI": (39.18, -76.67, -.05), "SLC": (40.79, -111.98, -.20),
            "SAN": (32.73, -117.19, -.05), "IAH": (29.98, -95.34, .08), "DCA": (38.85, -77.04, .10),
            "MDW": (41.79, -87.75, .18), "PDX": (45.59, -122.6, -.10), "STL": (38.75, -90.37, -.08),
            "HNL": (21.32, -157.9, -.30), "TPA": (27.98, -82.53, -.02), "AUS": (30.19, -97.67, .00)}


def _miles(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(h))


def generate(path="data/flights.csv", n=100_000, seed=42):
    rng = np.random.default_rng(seed)
    codes = list(AIRPORTS)
    al = rng.choice(list(AIRLINES), n)
    o = rng.choice(codes, n)
    d = rng.choice(codes, n)
    same = o == d
    d[same] = np.array(codes)[(np.array([codes.index(x) for x in d[same]]) + 7) % len(codes)]

    month = rng.integers(1, 13, n)
    dow = rng.integers(1, 8, n)
    dep_hour = np.clip(np.round(rng.normal(13, 4.2, n)), 5, 23).astype(int)
    crs = dep_hour * 100 + rng.choice([0, 15, 30, 45], n)
    dist = np.array([_miles(AIRPORTS[a][:2], AIRPORTS[b][:2]) for a, b in zip(o, d)]).round(0)

    winter = np.isin(month, [12, 1, 2])
    summer = np.isin(month, [6, 7, 8])
    precip = np.where(rng.random(n) < np.where(winter, .30, np.where(summer, .20, .15)),
                      rng.exponential(4, n), 0).round(1)
    wind = np.clip(rng.gamma(4, 2.5, n) + 3 * winter, 0, 60).round(0)
    vis = np.clip(10 - 4 * (precip > 0) * rng.random(n) - 3 * (rng.random(n) < .05), 0.25, 10).round(1)
    temp = (55 + 22 * np.cos((month - 7) * np.pi / 6 * -1) + rng.normal(0, 8, n)).round(0)

    logit = (-2.55 + np.array([AIRLINES[a] for a in al]) + np.array([AIRPORTS[a][2] for a in o])
             + 0.085 * np.clip(dep_hour - 6, 0, 14)
             + 0.40 * summer + 0.30 * winter + 0.15 * np.isin(dow, [5, 7])
             + 0.55 * (precip > 0) + 0.04 * np.minimum(precip, 15)
             + 0.35 * (vis < 3) + 0.30 * (wind > 25) + 0.25 * (temp < 25)
             - 0.0001 * dist + rng.normal(0, 0.35, n))
    delayed = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({"Reporting_Airline": al, "Origin": o, "Dest": d, "Month": month,
                       "DayOfWeek": dow, "CRSDepTime": crs, "Distance": dist,
                       "precip_mm": precip, "wind_kts": wind, "visibility_mi": vis,
                       "temp_f": temp, "ArrDel15": delayed})
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = generate()
    print("saved data/flights.csv", d.shape, "| delay rate:", round(d.ArrDel15.mean(), 3))
