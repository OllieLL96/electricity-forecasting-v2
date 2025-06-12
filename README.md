# ⚡ UK Electricity Demand Forecasting App

A real-time electricity demand forecasting tool built for energy market simulation, designed with traders and grid operators in mind.

## 🔍 Overview

This interactive Streamlit app forecasts hourly UK electricity demand using weather, calendar, and time-based features. It supports multiple ML models, stress testing, and dynamic forecast evaluation with LLM-generated insights.

**Key Features:**
- 📈 Forecast actual vs predicted electricity load (6–24 hour window)
- 🔁 Choose between XGBoost, Random Forest, and Linear Regression
- 🌦 Incorporates temperature, weekday/weekend, holiday & lag features
- 🌀 Recursive forecasting simulation with stress test mode
- ⚠️ Forecast error & imbalance risk flags
- 🔌 Peak load detection & flex response recommendations
- 🧠 GPT-powered natural language insights on forecast performance

## 📊 Why I Built This

Grid flexibility is essential in a decarbonised energy system. I built this tool to explore:
- Intra-day demand forecasting under changing weather or market conditions
- How small forecasting errors could lead to imbalance risk during peak demand
- How traders might simulate operational responses (e.g., dispatching storage)

## 💡 Technologies Used

- Python, Pandas, Plotly, Streamlit
- Scikit-learn, XGBoost
- OpenAI API (for GPT-3.5 insights)

## 🚀 Try It Out

👉 [Launch the App](https://electricity-forecasting-v2.streamlit.app/)

## 📂 Files

- `main.py` — Streamlit app logic
- `uk_demand_data.csv` — Historical UK load + weather feature set
- `.streamlit/secrets.toml` — API keys (private)

---
