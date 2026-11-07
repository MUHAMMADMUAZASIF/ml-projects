# 🚗 Traffic Forecaster
Predicts highway speed 15 / 30 / 60 minutes ahead for every road sensor (LightGBM).

## Run locally
```bash
pip install -r requirements.txt
python generate_data.py      # creates data/traffic_speeds.csv (demo data)
python train.py              # trains 3 models, prints metrics, writes outputs/metrics.json
```
## Run in Google Colab
Upload `traffic-forecaster_colab.ipynb` to https://colab.research.google.com and click **Runtime ▸ Run all**.

## Use the REAL METR-LA data
Download `metr-la.h5` (DCRNN project) and convert it to the CSV layout this project expects:
```python
import pandas as pd
df = pd.read_hdf("metr-la.h5")          # index = timestamps, columns = sensor ids
df = df.replace(0, float("nan")).ffill().bfill()   # 0 means missing in METR-LA
df.index.name = "timestamp"
df.to_csv("data/traffic_speeds.csv")
```
Then `python train.py`. (METR-LA speeds are in mph, 5-minute steps - same as the demo.)
