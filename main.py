# Author: Prof. Pedram Jahangiry
# Date: 2024-10-10

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sklearn.metrics import mean_squared_error, mean_absolute_error

def manual_train_test_split(y, train_size):
    split_point = int(len(y) * train_size)
    return y[:split_point], y[split_point:]

def run_forecast(y_train, y_test, model, fh, **kwargs):
    if model == 'ETS':
        forecaster = AutoETS(**kwargs)
    elif model == 'ARIMA':
        forecaster = AutoARIMA(**kwargs)
    else:
        raise ValueError("Unsupported model")
    
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index.drop_duplicates(), is_relative=False))

    last_date = y_test.index[-1]
    future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq).drop_duplicates()
    future_horizon = ForecastingHorizon(future_dates, is_relative=False)
    y_forecast = forecaster.predict(fh=future_horizon)
    
    return forecaster, y_pred, y_forecast

def plot_time_series(y_train, y_test, y_pred, y_forecast, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_train.index.to_timestamp(), y_train.values, label="Train")
    ax.plot(y_test.index.to_timestamp(), y_test.values, label="Test")
    ax.plot(y_pred.index.to_timestamp(), y_pred.values, label="Test Predictions")
    ax.plot(y_forecast.index.to_timestamp(), y_forecast.values, label="Forecast")
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("Time Series Forecasting App")

    col1, col2, col3 = st.columns([1.5, 3.5, 5])

    with col1:
        st.header("Model Assumptions")
        model_choice = st.selectbox("Select a model", ["ETS", "ARIMA"])
        train_size = st.slider("Train size (%)", 50, 95, 80) / 100

        if model_choice == "ETS":
            error = st.selectbox("Error type", ["add", "mul"])
            trend = st.selectbox("Trend type", ["add", "mul", None])
            seasonal = st.selectbox("Seasonal type", ["add", "mul", None])
            damped_trend = st.checkbox("Damped trend", value=False)
            seasonal_periods = st.number_input("Seasonal periods", min_value=1, value=1)
            model_params = {
                "error": error,
                "trend": trend,
                "seasonal": seasonal,
                "damped_trend": damped_trend,
                "sp": seasonal_periods,
            }
        elif model_choice == "ARIMA":
            st.subheader("Non-seasonal")
            start_p = st.number_input("Min p", min_value=0, value=0)
            max_p = st.number_input("Max p", min_value=0, value=5)
            start_q = st.number_input("Min q", min_value=0, value=0)
            max_q = st.number_input("Max q", min_value=0, value=5)
            d = st.number_input("d", min_value=0, value=1)

            st.subheader("Seasonal")
            seasonal = st.checkbox("Seasonal", value=True)
            if seasonal:
                start_P = st.number_input("Min P", min_value=0, value=0)
                max_P = st.number_input("Max P", min_value=0, value=2)
                start_Q = st.number_input("Min Q", min_value=0, value=0)
                max_Q = st.number_input("Max Q", min_value=0, value=2)
                D = st.number_input("D", min_value=0, value=1)
                sp = st.number_input("Periods", min_value=1, value=12)

            model_params = {
                "start_p": start_p,
                "max_p": max_p,
                "start_q": start_q,
                "max_q": max_q,
                "d": d,
                "seasonal": seasonal,
            }
            if seasonal:
                model_params.update({
                    "start_P": start_P,
                    "max_P": max_P,
                    "start_Q": start_Q,
                    "max_Q": max_Q,
                    "D": D,
                    "sp": sp
                })

    with col2:
        st.header("Data Handling")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                datetime_candidates = [
                    col for col in df.columns
                    if (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]))
                    and pd.to_datetime(df[col], errors='coerce').notna().sum() > 0
                ]

                if not datetime_candidates:
                    st.error("No valid datetime-like columns found.")
                else:
                    datetime_column = st.selectbox("Select the datetime column", datetime_candidates)

                    df['datetime'] = pd.to_datetime(df[datetime_column], errors='coerce')
                    df = df.dropna(subset=['datetime']).drop_duplicates(subset='datetime').sort_values('datetime')
                    df = df.set_index('datetime')

                    inferred_freq = pd.infer_freq(df.index)
                    freq_options = ['H', 'D', 'W', 'M', 'Q', 'Y']

                    if inferred_freq:
                        st.info(f"Detected frequency: **{inferred_freq}**")
                        index_to_use = freq_options.index(inferred_freq) if inferred_freq in freq_options else 0
                        freq = st.selectbox("Confirm or change frequency", freq_options, index=index_to_use)
                    else:
                        st.warning("Could not infer frequency. Please select manually.")
                        freq = st.selectbox("Select the data frequency", freq_options)

                    try:
                        df.index = df.index.to_period(freq)
                        df = df.loc[df.index.notnull()]
                    except Exception as e:
                        st.error(f"Unable to convert index: {str(e)}")

                    st.subheader("Data Preview")
                    st.write(df.head())

                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    if not numeric_columns:
                        st.error("No numeric columns found.")
                    else:
                        target_variable = st.selectbox("Select your target variable", numeric_columns)
                        st.subheader(f"Time Series Plot: {target_variable}")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(df.index.to_timestamp(), df[target_variable])
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")

    with col3:
        st.header("Forecast Results")
        fh = st.number_input("Number of periods to forecast", min_value=1, value=10)
        run_forecast_button = st.button("Run Forecast")

        if run_forecast_button:
            if 'df' in locals() and 'target_variable' in locals():
                try:
                    y = df[target_variable]
                    y_train, y_test = manual_train_test_split(y, train_size)
                    forecaster, y_pred, y_forecast = run_forecast(y_train, y_test, model_choice, fh, **model_params)

                    fig = plot_time_series(y_train, y_test, y_pred, y_forecast, f"{model_choice} Forecast for {target_variable}")
                    st.pyplot(fig)

                    st.subheader("Test Set Predictions")
                    st.write(y_pred)

                    st.subheader("Future Forecast Values")
                    st.write(y_forecast)

                    # Naïve baseline
                    naive_pred = pd.Series(y_train.iloc[-1], index=y_test.index)
                    naive_mse = mean_squared_error(y_test, naive_pred)
                    naive_rmse = np.sqrt(naive_mse)
                    naive_mae = mean_absolute_error(y_test, naive_pred)

                    st.subheader("Naïve Baseline Performance")
                    st.metric("Naïve RMSE", round(naive_rmse, 2))
                    st.metric("Naïve MAE", round(naive_mae, 2))

                    with st.expander("Model Summary"):
                        st.text(forecaster.summary())

                    csv = y_forecast.to_frame(name="Forecast").reset_index()
                    csv.columns = ["Date", "Forecast"]
                    csv_data = csv.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Forecast CSV",
                        data=csv_data,
                        file_name="forecast_results.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"An error occurred during forecasting: {str(e)}")
            else:
                st.warning("Please upload data and select a target variable before running the forecast.")

if __name__ == "__main__":
    main()
