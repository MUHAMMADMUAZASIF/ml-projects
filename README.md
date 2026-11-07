# 3 Machine Learning Projects
| Project | Model | Task |
|---|---|---|
| `traffic-forecaster/` | LightGBM | Predict highway speeds 15/30/60 min ahead |
| `airbnb-pricing/` | XGBoost + SHAP + Streamlit | Predict nightly price, explain drivers |
| `flight-delay/` | LightGBM + Streamlit | Predict probability a flight is 15+ min late |

Each folder has: `generate_data.py`, `data/*.csv`, `train.py`, a Colab notebook (`*_colab.ipynb`), `requirements.txt`, `README.md`.
See `RESULTS.md` for accuracy / precision / recall / F1.

## Open in Colab after uploading to GitHub
Replace `YOUR_USERNAME` and `YOUR_REPO`:
`https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/flight-delay/flight-delay_colab.ipynb`
