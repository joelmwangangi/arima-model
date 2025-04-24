import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

st.set_page_config(page_title="ARIMA Model GUI", layout="wide")
st.title("ARIMA Model GUI")

# 1) Configure your data source URL
DATA_URL = "https://github.com/joelmwangangi/arima-model/blob/1d871b0560e6dbf54779f8db3df2ca3dce736b4d/aapl_1y.csv"

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    # Load without parse_dates
    df = pd.read_csv(url)
    # Detect and normalize a date column
    for col in ("Date", "date", "timestamp", "Datetime"):
        if col in df.columns:
            df["Date"] = pd.to_datetime(df[col])
            break
    else:
        st.error(
            "No date column found in the CSV. "
            "Expected one of: Date, date, timestamp, Datetime."
        )
        st.stop()
    return df

# 2) Automatically load the data
try:
    df = load_data(DATA_URL)
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Could not load data from URL:\n{e}")
    st.stop()

# 3) Show data preview
st.subheader("Data Preview")
st.dataframe(df.head())

# Sidebar: model controls
st.sidebar.header("Model Settings")

order = st.sidebar.text_input(
    "ARIMA order (p, d, q)",
    value="1,1,1",
    help="Enter three integers separated by commas, e.g. 1,1,1"
)

fit_button = st.sidebar.button("Fit ARIMA Model")
results = None
if fit_button:
    if "Close" not in df.columns:
        st.sidebar.error("No 'Close' column found in the data.")
    else:
        try:
            p, d, q = map(int, order.split(","))
            series = df["Close"]
            model = ARIMA(series, order=(p, d, q))
            results = model.fit()
            st.sidebar.success(f"Model fitted! AIC: {results.aic:.2f}")
        except Exception as e:
            st.sidebar.error(f"Failed to fit model: {e}")

# 4) Plot time series
if st.sidebar.checkbox("Show Time Series Plot"):
    if "Close" not in df.columns:
        st.warning("No 'Close' column found to plot.")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["Date"], df["Close"], label="Close Price")
        ax.set_title("Close Price Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.grid(True)
        st.pyplot(fig)

# 5) Display summary & forecast
if results is not None:
    st.subheader("Model Summary")
    st.text(results.summary().as_text())

    st.subheader("Forecast")
    steps = st.number_input("Forecast steps", min_value=1, max_value=100, value=10)
    if st.button("Generate Forecast"):
        fc = results.forecast(steps=steps)
        st.line_chart(fc)
        st.dataframe(pd.DataFrame({"Forecast": fc}))
