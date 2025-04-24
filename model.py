import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Title and description
st.title("ARIMA Model for Time Series Analysis")
st.write("This application demonstrates ARIMA modeling and forecasting using your dataset.")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("aapl_1y.csv", parse_dates=["Date"], index_col="Date")

data = load_data()

# Show dataset preview
st.subheader("Dataset Preview")
st.write(data.head())

# Ensure 'Close' column is in the data for ARIMA modeling
if "Close" not in data.columns:
    st.error("The dataset must contain a 'Close' column for ARIMA modeling.")
else:
    # Prepare data for ARIMA
    time_series = data["Close"]

    # Model order selection
    st.sidebar.subheader("ARIMA Order Selection")
    p = st.sidebar.number_input("AR Order (p)", min_value=0, max_value=5, value=1, step=1)
    d = st.sidebar.number_input("Difference Order (d)", min_value=0, max_value=5, value=1, step=1)
    q = st.sidebar.number_input("MA Order (q)", min_value=0, max_value=5, value=0, step=1)

    # Fit ARIMA model
    model = ARIMA(time_series, order=(p, d, q))
    model_fit = model.fit()

    # Display model summary
    st.subheader("ARIMA Model Summary")
    st.text(model_fit.summary())

    # Forecasting
    forecast_steps = st.sidebar.number_input("Forecast Steps", min_value=1, max_value=30, value=10, step=1)
    forecast = model_fit.forecast(steps=forecast_steps)

    # Plot forecast
    st.subheader("Forecast Plot")
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, label="Original Data")
    plt.plot(forecast, label="Forecast", color="orange")
    plt.legend()
    st.pyplot(plt)
