# 🏠 Airbnb Pricing Engine
Predicts a fair nightly price, explains which features add the most money (SHAP), and a Streamlit app tells hosts what they're leaving on the table. Model: XGBoost.

## Run locally
```bash
pip install -r requirements.txt
python generate_data.py      # demo data -> data/airbnb_listings.csv
python train.py              # trains + writes outputs/model.joblib, metrics.json, shap_summary.png
streamlit run app.py         # opens the web app
```
## Run in Google Colab
Upload `airbnb-pricing_colab.ipynb` to https://colab.research.google.com ▸ **Run all**.

## Use REAL Inside Airbnb data
1. Download `listings.csv.gz` (the *detailed* file) for a city from http://insideairbnb.com/get-the-data/
2. `python prepare_real_data.py listings.csv.gz data/airbnb_listings.csv`
3. `python train.py`  (column names occasionally change between releases - edit `prepare_real_data.py` if needed)
