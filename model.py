import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Title and Description
st.title("ARIMA Model for Apple Stock Data Analysis")
st.write("This application processes the provided dataset, builds an ARIMA model, and forecasts future values.")

# Load the dataset
@st.cache_data
def load_data():
    # Reading the CSV file
    return pd.read_csv("aapl_1y.csv", parse_dates=["Date"], index_col="Date")

data = load_data()

# Filter data for 2023 onwards
data = data.loc[data.index >= pd.Timestamp("2023-01-01")]

# Display dataset preview
st.subheader("Dataset Preview")
st.write(data.head())

# Check for 'Close' column
if "Close" not in data.columns:
    st.error("The dataset must have a 'Close' column for ARIMA modeling.")
else:
    # Time series for ARIMA
    time_series = data["Close"]

    # Sidebar for ARIMA order selection
    st.sidebar.subheader("ARIMA Order Selection")
    p = st.sidebar.number_input("AR Order (p)", min_value=0, max_value=5, value=1, step=1)
    d = st.sidebar.number_input("Difference Order (d)", min_value=0, max_value=5, value=1, step=1)
    q = st.sidebar.number_input("MA Order (q)", min_value=0, max_value=5, value=0, step=1)

    # Fit ARIMA model
    model = ARIMA(time_series, order=(p, d, q))
    model_fit = model.fit()

    # Model summary
    st.subheader("ARIMA Model Summary")
    st.text(model_fit.summary())

    # Forecasting
    forecast_steps = st.sidebar.number_input("Forecast Steps", min_value=1, max_value=30, value=10, step=1)
    forecast = model_fit.forecast(steps=forecast_steps)

    # Forecast plot
    st.subheader("Forecast Plot")
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, label="Original Data")
    plt.plot(pd.date_range(start=time_series.index[-1], periods=forecast_steps + 1, freq='B')[1:], 
             forecast, label="Forecast", color="orange")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("ARIMA Forecast")
    st.pyplot(plt)
