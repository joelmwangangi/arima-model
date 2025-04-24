import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

# Function to load the dataset
def load_file():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load the CSV file into a pandas DataFrame
        dataset = pd.read_csv(uploaded_file, parse_dates=["Date"])
        st.write("Data preview:", dataset.head())
        return dataset
    return None

# Function to run the ARIMA model
def run_arima(dataset, order):
    if dataset is not None:
        try:
            # Assuming 'Close' column is relevant
            data_series = dataset["Close"]
            order = tuple(map(int, order.split(",")))
            
            # Fit ARIMA model
            model = ARIMA(data_series, order=order)
            results = model.fit()

            # Display ARIMA results summary
            st.write(f"ARIMA Model Summary:")
            st.write(f"AIC: {results.aic:.2f}")
            st.write(f"Model Summary: {results.summary()}")
            
            return results
        except Exception as e:
            st.error(f"An error occurred while fitting the ARIMA model: {e}")
    else:
        st.warning("No dataset loaded!")
        return None

# Function to plot the data
def plot_data(dataset):
    if dataset is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(dataset["Date"], dataset["Close"], label="Close Price")
        plt.title("Close Price Over Time")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

# Streamlit UI
st.title("ARIMA Model Deployment")

# Automatically load the data
dataset = pd.read_csv('aapl_1y.csv', parse_dates=["Date"])
st.write("Data Loaded Automatically from aapl_1y.csv")

# Show data preview
st.write(dataset.head())

# Step 1: Enter ARIMA order (p, d, q)
order = st.text_input("Enter ARIMA order (p, d, q)", "1,1,1")

# Step 2: Run the ARIMA model
if st.button("Run ARIMA Model"):
    results = run_arima(dataset, order)

# Step 3: Plot data
if st.button("Plot Data"):
    plot_data(dataset)

# Step 4: Show forecast (optional)
if results is not None:
    st.write("Model Forecast:")
    forecast_steps = st.slider("How many steps to forecast?", 1, 50, 10)
    forecast = results.forecast(steps=forecast_steps)
    st.write(f"Forecast for next {forecast_steps} steps:", forecast)
