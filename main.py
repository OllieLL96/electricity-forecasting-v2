import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta

# --- Helper: Recursive Forecasting ---
def recursive_forecast(df, model, features, start_ts, forecast_range, stress_test=False):
    df_forecast = df.copy()
    preds = []
    timestamps = pd.date_range(start=start_ts, periods=forecast_range, freq="H", tz="Europe/London")

    for t in timestamps:
        row = df_forecast.loc[t - pd.Timedelta(hours=1)].copy()

        lag_24h = df_forecast.loc[t - pd.Timedelta(hours=24), "load"] if (t - pd.Timedelta(hours=24)) in df_forecast.index else preds[-24] if len(preds) >= 24 else np.nan
        lag_168h = df_forecast.loc[t - pd.Timedelta(hours=168), "load"] if (t - pd.Timedelta(hours=168)) in df_forecast.index else preds[-168] if len(preds) >= 168 else np.nan

        if stress_test:
            if not pd.isnull(lag_24h): lag_24h *= np.random.normal(1, 0.05)
            if not pd.isnull(lag_168h): lag_168h *= np.random.normal(1, 0.05)
            row["temp_C"] -= np.random.choice([0, 2, 4])

        feature_row = {
            "temp_C": row["temp_C"],
            "hour": t.hour,
            "dayofweek": t.dayofweek,
            "is_weekend": t.weekday() >= 5,
            "is_peak_hour": t.hour in [7, 8, 17, 18, 19],
            "is_holiday": row.get("is_holiday", False),
            "lag_24h": lag_24h,
            "lag_168h": lag_168h
        }

        if np.any(pd.isnull(list(feature_row.values()))):
            preds.append(np.nan)
            continue

        X = pd.DataFrame([feature_row])
        pred = model.predict(X)[0]
        preds.append(pred)
        df_forecast.loc[t, "load"] = pred

    return pd.DataFrame({"timestamp": timestamps, "Predicted Load": preds}).set_index("timestamp")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("uk_demand_data.csv", parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

# --- Train models ---
@st.cache_resource
def train_models():
    df = load_data()
    features = ["temp_C", "hour", "dayofweek", "is_weekend", "is_peak_hour", "is_holiday", "lag_24h", "lag_168h"]
    X = df[features]
    y = df["load"]

    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X, y)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    lr_model = LinearRegression()
    lr_model.fit(X, y)

    return {
        "XGBoost": xgb_model,
        "Random Forest": rf_model,
        "Linear Regression": lr_model
    }

# --- App UI ---
st.set_page_config(page_title="Electricity Forecasting App", layout="wide")
st.title("⚡ UK Electricity Demand Forecasting with Weather & Peak Risk")

st.markdown("""
This tool demonstrates a real-time electricity demand forecast model built with multiple models:
- **XGBoost** (default, boosted trees)
- **Random Forest**
- **Linear Regression**

It leverages historical demand, weather, and calendar effects to simulate intra-day forecasting.
""")

df = load_data()
models = train_models()
features = ["temp_C", "hour", "dayofweek", "is_weekend", "is_peak_hour", "is_holiday", "lag_24h", "lag_168h"]

# Sidebar
st.sidebar.header("🔧 Forecast Settings")
model_choice = st.sidebar.selectbox("Choose prediction model:", list(models.keys()))
selected_model = models[model_choice]

valid_dates = df.index.normalize().unique()
selected_date = st.sidebar.date_input("Select a date to simulate forecast:", value=valid_dates[-2], min_value=valid_dates[0], max_value=valid_dates[-2])
forecast_range = st.sidebar.slider("Forecast range (hours):", 6, 24, 24)
use_recursive = st.sidebar.checkbox("🔁 Use Recursive Forecasting", value=False)
stress_test = st.sidebar.checkbox("🔥 Enable Stress Test Mode", value=False)

# Data slice
start_ts = pd.Timestamp(selected_date, tz="Europe/London")
end_ts = start_ts + timedelta(hours=forecast_range - 1)
df_day = df.loc[start_ts:end_ts].copy()

if df_day.empty:
    st.warning("No data available for this date.")
else:
    if use_recursive:
        st.markdown(f"🌀 **Using recursive forecasting simulation ({model_choice})...**")
        df_preds = recursive_forecast(df, selected_model, features, start_ts, forecast_range, stress_test=stress_test)
        df_day["Predicted Load"] = df_preds["Predicted Load"]
    else:
        st.markdown(f"✅ **Using direct prediction from known inputs ({model_choice})...**")
        X_day = df_day[features]
        df_day["Predicted Load"] = selected_model.predict(X_day)

    y_true = df_day["load"]
    df_day["Error"] = df_day["Predicted Load"] - y_true
    df_day["Error %"] = df_day["Error"] / y_true * 100
    df_day["Imbalance Risk"] = df_day["Error %"].abs() > 5

    # Peak load marker
    threshold = df["load"].quantile(0.90)
    df_day["Peak Load"] = df_day["load"] >= threshold

    # Visuals
    st.subheader(f"📈 Actual vs Predicted Load ({model_choice})")
    fig1 = px.line(df_day, x=df_day.index, y=["load", "Predicted Load"], labels={"value": "MW", "timestamp": "Time"})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("⚠️ Forecast Error (%) with Imbalance Risk")
    fig2 = px.scatter(df_day, x=df_day.index, y="Error %", color="Imbalance Risk",
                      color_discrete_map={True: "red", False: "blue"}, labels={"Error %": "Forecast Error (%)"})
    fig2.add_hline(y=5, line_dash="dot", line_color="gray")
    fig2.add_hline(y=-5, line_dash="dot", line_color="gray")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 Peak Load Detection")
    fig3 = px.line(df_day, x=df_day.index, y="load", title="Load with Peak Markers")
    fig3.add_scatter(x=df_day[df_day["Peak Load"]].index,
                     y=df_day[df_day["Peak Load"]]["load"],
                     mode="markers", marker=dict(color="orange", size=6), name="Peak Load")
    st.plotly_chart(fig3, use_container_width=True)

    # Performance
    st.subheader("📋 Forecast Performance")
    rmse = np.sqrt(mean_squared_error(y_true, df_day["Predicted Load"]))
    mae = mean_absolute_error(y_true, df_day["Predicted Load"])
    st.markdown(f"**Model:** {model_choice}  |  **RMSE:** {rmse:.2f} MW  |  **MAE:** {mae:.2f} MW")

    # Insights
    st.subheader("🧠 Insights from This Forecast")
    risky_hours = df_day[df_day["Imbalance Risk"]].index.hour.unique().tolist()
    if risky_hours:
        st.success(f"{len(risky_hours)} hours had forecast error > 5%.")
        st.write("These occurred at hours:", risky_hours)
        if df_day["Peak Load"].any():
            st.info("⚡ Some of these hours overlapped with peak load — potential for imbalance risk.")
    else:
        st.info("✅ No high-risk forecast errors detected in this window.")

    # Flex placeholder
    st.subheader("🔌 Example Flex Response")
    if df_day["Imbalance Risk"].any():
        st.markdown("""
        Based on the imbalance risk detected, an operator could:
        - Dispatch **battery storage** during the most error-prone hour
        - Shift EV charging or reduce load to reduce cost exposure
        """)
    else:
        st.write("No corrective flex action needed in this interval.")
    # 💬 LLM Forecast Insight Section
st.subheader("💬 Ask the AI about this forecast")

user_question = st.text_area(
    "Ask a question or request a summary of the forecast:",
    placeholder="e.g. What risks are present in this forecast window?"
)

if st.button("Generate Insight"):
    import openai

    try:
        openai.api_key = st.secrets["openai"]["api_key"]

        # Prepare last 24-hour slice for context
        context_summary = df_day[["Predicted Load", "load", "Error %", "Imbalance Risk", "Peak Load"]].tail(24).to_csv()

        system_prompt = (
            "You are a forecasting assistant helping a UK energy trader assess electricity demand. "
            "Interpret the user's question using the forecast data, highlight risk, imbalance potential, and any trading or grid flexibility insights."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Data:\n{context_summary}\n\nQuestion:\n{user_question}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3,
        )
        st.success(response["choices"][0]["message"]["content"])

    except Exception as e:
        st.error(f"LLM call failed: {e}")    

    # Why built
    with st.expander("📘 Why I Built This Tool"):
        st.markdown("""
        I created this app to simulate the real-world challenges of intra-day demand forecasting.

        By integrating weather data, engineered lag features, and real-time forecast simulation,
        I wanted to explore:
        - How small errors can escalate during peak demand
        - How operational risk flags could help traders and grid operators make better decisions
        - How tools like this might support Octopus' vision of a smart, flexible, decarbonised grid
        """)
