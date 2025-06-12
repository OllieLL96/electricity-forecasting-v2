import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Load data and model ---
@st.cache_data
def load_data():
    df = pd.read_csv("uk_demand_data.csv", parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    df = load_data()
    features = ["temp_C", "hour", "dayofweek", "is_weekend", "is_peak_hour", "is_holiday", "lag_24h", "lag_168h"]
    X = df[features]
    y = df["load"]
    model.fit(X, y)
    return model

# --- App UI ---
st.set_page_config(page_title="Electricity Forecasting App", layout="wide")
st.title("⚡ UK Electricity Demand Forecasting with Weather & Peak Risk")

st.markdown("""
This tool demonstrates a real-time electricity demand forecast model built with **XGBoost**, leveraging:
- Historical demand data
- Weather (temperature)
- Time-based features (hour, weekday, holiday)

It highlights forecast accuracy, peak load risk, and model explainability — designed with energy trading & grid ops in mind.
""")

# --- Load data ---
df = load_data()
model = load_model()
features = ["temp_C", "hour", "dayofweek", "is_weekend", "is_peak_hour", "is_holiday", "lag_24h", "lag_168h"]

# --- Sidebar: Forecast Controls ---
st.sidebar.header("🔧 Forecast Settings")
valid_dates = df.index.normalize().unique()
selected_date = st.sidebar.date_input("Select a date to simulate forecast:", value=valid_dates[-2], min_value=valid_dates[0], max_value=valid_dates[-2])
forecast_range = st.sidebar.slider("Forecast range (hours):", 6, 24, 24)

# --- Filter & Prepare Input ---
start_ts = pd.Timestamp(selected_date, tz="Europe/London")
end_ts = start_ts + timedelta(hours=forecast_range - 1)
df_day = df.loc[start_ts:end_ts].copy()

if df_day.empty:
    st.warning("No data available for this date.")
else:
    # --- Predict ---
    X_day = df_day[features]
    y_true = df_day["load"]
    y_pred = model.predict(X_day)

    df_day["Predicted Load"] = y_pred
    df_day["Error"] = y_pred - y_true
    df_day["Error %"] = df_day["Error"] / y_true * 100
    df_day["Imbalance Risk"] = df_day["Error %"].abs() > 5

    # --- Visual: Actual vs Predicted ---
    st.subheader("📈 Actual vs Predicted Load")
    fig1 = px.line(df_day, x=df_day.index, y=["load", "Predicted Load"], labels={"value": "MW", "timestamp": "Time"})
    st.plotly_chart(fig1, use_container_width=True)

    # --- Visual: Forecast Error with Risk Flag ---
    st.subheader("⚠️ Forecast Error (%) with Imbalance Risk")
    fig2 = px.scatter(df_day, x=df_day.index, y="Error %", color="Imbalance Risk",
                      color_discrete_map={True: "red", False: "blue"}, labels={"Error %": "Forecast Error (%)"})
    fig2.add_hline(y=5, line_dash="dot", line_color="gray")
    fig2.add_hline(y=-5, line_dash="dot", line_color="gray")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Visual: Highlight Peak Load Hours ---
    st.subheader("📊 Peak Load Detection")
    threshold = df["load"].quantile(0.90)
    df_day["Peak Load"] = df_day["load"] >= threshold
    fig3 = px.line(df_day, x=df_day.index, y="load", title="Load with Peak Markers")
    fig3.add_scatter(x=df_day[df_day["Peak Load"]].index,
                     y=df_day[df_day["Peak Load"]]["load"],
                     mode="markers", marker=dict(color="orange", size=6), name="Peak Load")
    st.plotly_chart(fig3, use_container_width=True)

    # --- Summary Metrics ---
    st.subheader("📋 Forecast Performance")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    st.markdown(f"**RMSE:** {rmse:.2f} MW  |  **MAE:** {mae:.2f} MW")

    # --- Explanatory Notes ---
    with st.expander("ℹ️ What is this app doing?"):
        st.markdown("""
        - Trains a short-term XGBoost model on the most recent 35 days of hourly UK load
        - Uses lag features, weather, and time context to predict future load
        - Identifies risky forecast hours where the model may underperform (>5% error)
        - Flags high-demand (peak) hours as additional operational risk
        """)
