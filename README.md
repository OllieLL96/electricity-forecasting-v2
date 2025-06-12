# ⚡ Electricity Demand Forecasting with Weather, Risk & Peak Load Context

This Streamlit app demonstrates a real-time electricity demand forecasting tool trained on UK data, enriched with weather inputs and logic to identify peak load and imbalance risk — designed with energy trading and grid operations in mind.

## 🔍 What It Does

* **Forecasts hourly UK electricity demand** using an XGBoost regression model
* Incorporates:

  * Historical demand trends (lags)
  * Weather data (temperature)
  * Time-of-day, weekend, and holiday context
* Simulates **real-time predictions** with user-selected dates
* Visualizes:

  * Actual vs Predicted Load
  * Forecast error with imbalance risk flags
  * Peak demand hours (top 10%)

## 📊 Why It Matters

Forecasting electricity demand accurately is critical in:

* **Grid operations** — to avoid over/under-supply
* **Trading desks** — to optimize day-ahead and intraday positions
* **Imbalance markets** — where forecast errors become real financial penalties

This app showcases not just model performance but **when** errors happen, highlighting operational risk.

## 🚀 How to Use the App

1. Select a date from the past 35-day window
2. Choose how many hours to forecast (up to 24)
3. Explore:

   * Forecast accuracy (RMSE, MAE)
   * Visual error and risk flags
   * When peaks are expected (and whether you got them right)

## 🧠 Model Details

* **Model:** XGBoostRegressor
* **Features:**

  * Temperature (`temp_C`)
  * Hour, Day of Week
  * Weekend & Holiday flags
  * Lagged load (`lag_24h`, `lag_168h`)
  * Peak hour indicator
* **Data Source:** ENTSOE & Open-Meteo

## 📁 Files Included

* `main.py` – Streamlit app source code
* `uk_demand_data.csv` – Preprocessed demand + weather data
* `requirements.txt` – Dependencies for deployment


