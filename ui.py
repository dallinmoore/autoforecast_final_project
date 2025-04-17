import streamlit as st
import pandas as pd
import numpy as np
from models import get_model_params

def render_model_params_ui(model_choice):
    """
    Render UI components for model parameter selection
    
    Args:
        model_choice (str): Selected model name
        
    Returns:
        dict: Model parameters based on user selections
    """
    if model_choice == "ETS":
        error = st.selectbox("Error type", ["add", "mul"])
        trend = st.selectbox("Trend type", ["add", "mul", None])
        seasonal = st.selectbox("Seasonal type", ["add", "mul", None])
        damped_trend = st.checkbox("Damped trend", value=False)
        seasonal_periods = st.number_input("Seasonal periods", min_value=1, value=1)
        return {
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
        
        model_params = {
            "start_p": start_p,
            "max_p": max_p,
            "start_q": start_q,
            "max_q": max_q,
            "d": d,
            "seasonal": seasonal,
        }
        
        if seasonal:
            start_P = st.number_input("Min P", min_value=0, value=0)
            max_P = st.number_input("Max P", min_value=0, value=2)
            start_Q = st.number_input("Min Q", min_value=0, value=0)
            max_Q = st.number_input("Max Q", min_value=0, value=2)
            D = st.number_input("D", min_value=0, value=1)
            sp = st.number_input("Periods", min_value=1, value=12)
            
            model_params.update({
                "start_P": start_P,
                "max_P": max_P,
                "start_Q": start_Q,
                "max_Q": max_Q,
                "D": D,
                "sp": sp
            })
        
        return model_params
    
    elif model_choice == "RandomForest" or model_choice == "XGBoost":
        # Common ML parameters section
        st.subheader("Time Series Parameters")
        lags = st.number_input("Number of lags", min_value=1, value=3)  # Default to 3
        
        if model_choice == "RandomForest":
            st.subheader("Model Parameters")
            n_estimators = st.number_input("Number of trees", min_value=10, value=100)
            max_depth = st.number_input("Max depth", min_value=1, value=10)
            return {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": 42,
                "lags": lags  # Pass the lag parameter to the model
            }
        
        elif model_choice == "XGBoost":
            st.subheader("Model Parameters")
            learning_rate = st.number_input("Learning rate", min_value=0.01, value=0.1)
            n_estimators = st.number_input("Number of trees", min_value=10, value=100)
            max_depth = st.number_input("Max depth", min_value=1, value=10)
            return {
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": 42,
                "lags": lags  # Pass the lag parameter to the model
            }
    
    return {}

def render_data_upload_ui():
    """
    Render UI components for data upload and selection
    
    Returns:
        tuple: (df, target_variable, freq) - processed dataframe, selected target variable, and frequency
    """
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is None:
        return None, None, None
        
    try:
        # Get frequency selection
        freq_options = ['D', 'W', 'M', 'Q', 'Y']
        freq = st.selectbox("Select the data frequency", freq_options)
        
        # Import here to avoid circular imports
        from preprocess import load_and_preprocess_data, plot_original_series
        
        df = load_and_preprocess_data(uploaded_file, freq)
        
        st.subheader("Data Preview")
        st.write(df.head())

        # Filter out non-numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.error("No numeric columns found in the uploaded data. Please ensure your CSV contains numeric data for forecasting.")
            return None, None, freq
        
        target_variable = st.selectbox("Select your target variable", numeric_columns)

        # Plot the time series of the selected target variable
        st.subheader(f"Time Series Plot: {target_variable}")
        fig = plot_original_series(df, target_variable)
        st.pyplot(fig)
        
        return df, target_variable, freq
        
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
        st.error("Please ensure your CSV file is properly formatted with a date column and numeric data for forecasting.")
        return None, None, None