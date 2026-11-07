"""Generate a synthetic METR-LA-style traffic speed dataset (wide format).

Columns: timestamp, s00 ... sNN  (speed in mph, one row per 5 minutes).
This is the SAME layout you get from METR-LA / PeMS after converting to CSV,
so you can later swap in the real file without changing train.py.
"""
import numpy as np
import pandas as pd


def generate(path="data/traffic_speeds.csv", n_sensors=20, days=28, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-04", periods=days * 288, freq="5min")
    hour = idx.hour.values + idx.minute.values / 60
    weekday = (idx.dayofweek.values < 5).astype(float)

    base = rng.uniform(62, 70, n_sensors)            # free-flow speed
    am_sev = rng.uniform(0.15, 0.45, n_sensors)      # morning rush severity
    pm_sev = rng.uniform(0.15, 0.50, n_sensors)      # evening rush severity
    am = np.exp(-((hour - 8.0) / 1.0) ** 2)
    pm = np.exp(-((hour - 17.3) / 1.3) ** 2)
    midday = 0.25 * np.exp(-((hour - 13) / 2.5) ** 2)

    speeds = np.zeros((len(idx), n_sensors))
    for s in range(n_sensors):
        drop = weekday * (am_sev[s] * am + pm_sev[s] * pm) + (1 - weekday) * midday * am_sev[s]
        speeds[:, s] = base[s] * (1 - drop)

    # random incidents that propagate to upstream sensors with a delay
    for _ in range(days * 3):
        start = rng.integers(0, len(idx) - 60)
        dur = rng.integers(6, 18)
        s0 = rng.integers(0, n_sensors)
        mag = rng.uniform(0.25, 0.55)
        for k in range(4):
            s = s0 - k
            if s < 0:
                break
            a = start + k * 2
            b = min(a + dur + k * 3, len(idx))
            speeds[a:b, s] *= 1 - mag * (0.8 ** k)

    noise = pd.DataFrame(rng.normal(0, 1, speeds.shape)).ewm(alpha=0.4).mean().values * 2.5
    speeds = np.clip(speeds + noise, 5, 75).round(1)

    df = pd.DataFrame(speeds, index=idx, columns=[f"s{i:02d}" for i in range(n_sensors)])
    df.index.name = "timestamp"
    df.to_csv(path)
    return df


if __name__ == "__main__":
    d = generate()
    print("saved data/traffic_speeds.csv", d.shape)
