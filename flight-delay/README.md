# ✈️ Flight Delay Predictor
Predicts the probability that a US flight arrives 15+ minutes late (LightGBM) and serves it in a Streamlit page.

## Run locally
```bash
pip install -r requirements.txt
python generate_data.py      # demo data -> data/flights.csv
python train.py              # writes outputs/model.joblib + metrics.json
streamlit run app.py
```
## Run in Google Colab
Upload `flight-delay_colab.ipynb` to https://colab.research.google.com ▸ **Run all**.

## Use REAL BTS data
Download "Reporting Carrier On-Time Performance" CSVs from https://www.transtats.bts.gov (Download ▸ pick fields:
`Reporting_Airline, Origin, Dest, Month, DayOfWeek, CRSDepTime, Distance, ArrDel15`), concatenate months, save as `data/flights.csv`.
BTS has no weather: the weather columns are optional - `train.py` simply skips them if absent, or you can join
NOAA/Meteostat weather by Origin + date + hour to improve accuracy.
