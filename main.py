import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from preprocess import manual_train_test_split, plot_time_series
from models import run_forecast
from ui import render_model_params_ui, render_data_upload_ui, render_forecast_results_ui
from metrics import calculate_forecast_metrics, display_metrics, calculate_naive_baseline

def main():
    st.set_page_config(layout="wide")
    st.title("Time Series Forecasting App")

    col1, col2, col3 = st.columns([1.5, 3.5, 5])

    # Model selection and parameters (left column)
    with col1:
        st.header("Model Assumptions")
        model_choice = st.selectbox("Select a model", ["ETS", "ARIMA", "RandomForest", "XGBoost"])
        train_size = st.slider("Train size (%)", 50, 95, 80) / 100
        
        # Get model parameters from UI module
        model_params = render_model_params_ui(model_choice)

    # Data handling (middle column)
    with col2:
        st.header("Data Handling")
        df, target_variable, freq = render_data_upload_ui()

    # Forecast results (right column)
    with col3:
        st.header("Forecast Results")
        fh = st.number_input("Number of periods to forecast", min_value=1, value=10)
        run_forecast_button = st.button("Run Forecast")

        if run_forecast_button:
            if df is not None and target_variable is not None:
                try:
                    y = df[target_variable]
                    y_train, y_test = manual_train_test_split(y, train_size)
                    forecaster, y_pred, y_forecast = run_forecast(y_train, y_test, model_choice, fh, **model_params)

                    # Plot results

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=y_train.index.to_timestamp(), y=y_train.values, mode='lines', name='Train'))
                    fig.add_trace(go.Scatter(x=y_test.index.to_timestamp(), y=y_test.values, mode='lines', name='Test'))
                    fig.add_trace(go.Scatter(x=y_pred.index.to_timestamp(), y=y_pred.values, mode='lines', name='Test Predictions'))
                    fig.add_trace(go.Scatter(x=y_forecast.index.to_timestamp(), y=y_forecast.values, mode='lines', name='Forecast'))

                    fig.update_layout(
                        title=f"{model_choice} Forecast for {target_variable}",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        hovermode="x unified",
                        height=500,
                        width=1000
                    )

                    st.plotly_chart(fig, use_container_width=True)


                    # Calculate and display metrics
                    metrics = calculate_forecast_metrics(y_test, y_pred)
                    st.subheader("Forecast Accuracy Metrics")
                    st.table(display_metrics(metrics))

                    # Calculate and display naïve baseline
                    naive_metrics = calculate_naive_baseline(y_train, y_test)
                    st.subheader("Naïve Baseline Performance")
                    st.metric("Naïve RMSE", round(naive_metrics["RMSE"], 2))
                    st.metric("Naïve MAE", round(naive_metrics["MAE"], 2))
                    
                    # Display forecast results
                    render_forecast_results_ui(forecaster, y_pred, y_forecast)

                except Exception as e:
                    st.error(f"An error occurred during forecasting: {str(e)}")
            else:
                st.warning("Please upload data and select a target variable before running the forecast.")

if __name__ == "__main__":
    main()
