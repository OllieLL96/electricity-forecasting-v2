# ⚡ UK Electricity Demand Forecasting App

An interactive web app to simulate **intra-day electricity demand forecasting** — tailored for grid operators, energy traders, and forecasters. Built to support situational awareness, peak load management, and flexible asset deployment strategies.

## 🔍 Overview

This app forecasts UK electricity demand up to 24 hours ahead using weather and time-based features, with optional stress-testing for operational edge-case scenarios.

It was developed to simulate real-time trading environments, and is inspired by the kind of work done by **Octopus Energy Trading**.

## 🚀 Features

- 📈 Forecast vs. actual load visualization
- 🔁 Toggle between **XGBoost**, **Random Forest**, and **Linear Regression**
- 🔬 Recursive forecasting to simulate real-time predictions
- 🔥 Stress-testing for cold weather or lag drift shocks
- ⚠️ Imbalance risk detection based on forecast error thresholds
- 🔌 Peak load flagging & suggested operational response
- 💬 GPT-powered **natural language insights** from the latest forecast

---

## 📊 Data Sources

This app uses publicly available and open data sources:

| Data Type                  | Source                                               | Notes |
|---------------------------|------------------------------------------------------|-------|
| Electricity Load & Generation | ENTSO-E Transparency Platform                      | Bulk download (2015–2020) of historical data, via [https://transparency.entsoe.eu/](https://transparency.entsoe.eu/) |
| Weather (temperature)     | Open-Meteo Historical API                           | Hourly 2m temperature data for London |
| Calendar Holidays         | UK Government API                                   | Used to flag bank holidays in the forecast model |

> The raw dataset (`uk_demand_data.csv`) is a cleaned version prepared from the bulk download and merged with external weather and holiday features.

---

## 🧠 Models Used

You can toggle between 3 forecasting models:

- **XGBoost Regressor** (default): optimized tree boosting for tabular data
- **Random Forest Regressor**: ensemble-based for robustness
- **Linear Regression**: simple, interpretable baseline

Models are trained on engineered features including:
- Hour of day
- Day of week
- Public holidays
- Lag features (`t-24h`, `t-168h`)
- Temperature (°C)
- Weekend and peak hour flags

---

## 🌀 Recursive Forecasting & Stress Testing

You can simulate real-time conditions using recursive mode. When enabled:
- Lag features are updated hour by hour using prior forecasts
- Optional **stress test mode** simulates:
  - Cold weather shocks
  - Noise in lag-based memory

---

## 💬 LLM Forecast Insight Assistant

Using OpenAI’s GPT-3.5 API, the app can explain:
- Imbalance risks
- Peak overlaps
- Model deviations or anomaly flags

> **Prompt box is at the bottom of the page.** You can ask questions like:
> _"Summarise the risks from today's forecast"_ or _"What might explain the high errors at 6pm?"_

---

## 📂 Files

- `main.py` — Full app code
- `uk_demand_data.csv` — Pre-processed dataset (load + temp + features)
- `.streamlit/secrets.toml` — API key config for OpenAI

