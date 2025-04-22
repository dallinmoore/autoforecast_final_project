import streamlit as st
import pandas as pd
import numpy as np
from models import get_model_params

def render_step_navigation():
    """
    Render the step navigation UI
    
    Returns:
        int: Current step (1, 2, or 3)
    """
    # Check for step in session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    
    # Create step indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        step1_status = "active" if st.session_state.current_step == 1 else "complete" if st.session_state.current_step > 1 else "inactive"
        st.markdown(f"""
        <div class="step-indicator {step1_status}">
            <div class="step-number">1</div>
            <div class="step-title">Upload Data</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        step2_status = "active" if st.session_state.current_step == 2 else "complete" if st.session_state.current_step > 2 else "inactive"
        st.markdown(f"""
        <div class="step-indicator {step2_status}">
            <div class="step-number">2</div>
            <div class="step-title">Configure Model</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        step3_status = "active" if st.session_state.current_step == 3 else "inactive"
        st.markdown(f"""
        <div class="step-indicator {step3_status}">
            <div class="step-number">3</div>
            <div class="step-title">View Results</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add custom CSS for step indicators with improved styling
    st.markdown("""
    <style>
    .step-indicator {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .step-number {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .step-title {
        font-weight: 500;
    }
    .active {
        background-color: #1E90FF;
        color: white;
        border: 2px solid #1E90FF;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .complete {
        background-color: #28a745;
        color: white;
        border: 2px solid #28a745;
    }
    .inactive {
        background-color: #f8f9fa;
        color: #6c757d;
        border: 2px solid #e9ecef;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.session_state.current_step > 1:
            if st.button("⬅️ Previous Step"):
                st.session_state.current_step -= 1
                st.rerun()
    
    with col3:
        if st.session_state.current_step < 3:
            # Check if we can proceed to the next step
            can_proceed = True
            if st.session_state.current_step == 1:
                # Enable Next button as soon as a file is uploaded
                if '_uploaded_file' not in st.session_state:
                    can_proceed = False
                    
            if st.session_state.current_step == 2:
                # No confirmation button needed for Step 2 now
                can_proceed = True
                
            next_button = st.button("Next Step ➡️", disabled=not can_proceed)
            if next_button:
                # If moving from step 1 to 2, automatically save the data choices
                if st.session_state.current_step == 1:
                    st.session_state.data_uploaded = True
                # If moving from step 2 to 3, automatically save model configuration
                elif st.session_state.current_step == 2:
                    st.session_state.model_configured = True
                
                st.session_state.current_step += 1
                st.rerun()
    
    return st.session_state.current_step

def render_step1_data_upload():
    """
    Render step 1: Data upload and preprocessing
    
    Returns:
        tuple: (df, target_variable, freq) - processed dataframe, selected target variable, and frequency
    """
    st.header("Step 1: Upload Data")
    
    # Use session_state for the file uploader to maintain state
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
    
    # Set the flag as soon as a file is uploaded - directly check the file uploader value
    if uploaded_file is not None:
        # This is a safer way to track uploaded files
        st.session_state._uploaded_file = True
    else:
        st.session_state._uploaded_file = False
        st.info("Please upload a CSV file containing time series data")
        return None, None, None
    
    # Just for debugging - show if file was detected
    if st.session_state.get("file_debug"):
        st.success(st.session_state.file_debug)
        
    try:
        # Import here to avoid circular imports
        from preprocess import load_and_preprocess_data, plot_original_series
        
        # Initial loading of data to find datetime columns and infer frequency
        df = pd.read_csv(uploaded_file)
        
        # Find datetime columns
        datetime_candidates = [
            col for col in df.columns
            if (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]))
            and pd.to_datetime(df[col], errors='coerce').notna().sum() > 0
        ]
        
        if not datetime_candidates:
            st.error("No valid datetime-like columns found in the data.")
            return None, None, None
        
        # Let user select the datetime column
        datetime_column = st.selectbox("Select the datetime column", datetime_candidates)
        
        # Create a temporary datetime index to infer frequency
        temp_df = df.copy()
        temp_df['datetime'] = pd.to_datetime(temp_df[datetime_column], errors='coerce')
        temp_df = temp_df.dropna(subset=['datetime']).drop_duplicates(subset='datetime').sort_values('datetime')
        temp_df = temp_df.set_index('datetime')
        
        # Try to infer frequency
        inferred_freq = pd.infer_freq(temp_df.index)
        freq_options = ['h', 'D', 'W', 'M', 'Q', 'Y']
        
        # Standardize frequency - convert variations to standard forms
        if inferred_freq:
            # Convert 'MS' (month start) to 'M' (month end) and similar variations
            if inferred_freq == 'MS':
                inferred_freq = 'M'
                st.info(f"Detected frequency: **MS** (month start), using **M** (month) as standard frequency")
            elif inferred_freq == 'QS':
                inferred_freq = 'Q'
                st.info(f"Detected frequency: **QS** (quarter start), using **Q** (quarter) as standard frequency")
            elif inferred_freq == 'YS':
                inferred_freq = 'Y'
                st.info(f"Detected frequency: **YS** (year start), using **Y** (year) as standard frequency")
            else:
                st.info(f"Detected frequency: **{inferred_freq}**")
            
            # Try to use the detected frequency as default
            try:
                # Find the closest matching standard frequency
                if inferred_freq[0] in [f[0] for f in freq_options]:
                    # If the first character matches any standard frequency, use that
                    index_to_use = [f[0] for f in freq_options].index(inferred_freq[0])
                elif inferred_freq in freq_options:
                    index_to_use = freq_options.index(inferred_freq)
                else:
                    index_to_use = 0
                freq = st.selectbox("Confirm or change frequency", freq_options, index=index_to_use)
            except (ValueError, IndexError):
                st.warning(f"Detected frequency '{inferred_freq}' is not in standard options.")
                freq = st.selectbox("Select frequency", freq_options)
        else:
            st.warning("Could not infer frequency. Please select manually.")
            freq = st.selectbox("Select the data frequency", freq_options)
            
        # Now do the full preprocessing with the selected frequency
        try:
            # Reset the file pointer
            uploaded_file.seek(0)
            df, _, _ = load_and_preprocess_data(uploaded_file, freq)
            
            st.subheader("Data Preview")
            st.write(df.head())
            
            # Filter out non-numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                st.error("No numeric columns found in the uploaded data.")
                return None, None, freq
            
            target_variable = st.selectbox("Select your target variable", numeric_columns)
            
            # Plot the time series of the selected target variable with zoom and date filtering
            st.subheader(f"Time Series Plot: {target_variable}")

            # Allow user to select date range
            min_date = df.index.min().to_timestamp()
            max_date = df.index.max().to_timestamp()

            start_date, end_date = st.date_input(
                "Select date range for preview",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            # Filter data based on selected range
            filtered_df = df[(df.index.to_timestamp() >= pd.Timestamp(start_date)) &
                             (df.index.to_timestamp() <= pd.Timestamp(end_date))]

            # Convert index for Plotly
            plot_df = filtered_df.copy()
            plot_df.index = plot_df.index.to_timestamp()
            plot_df = plot_df.reset_index().rename(columns={plot_df.index.name: 'datetime'})

            import plotly.express as px
            fig = px.line(plot_df, x='datetime', y=target_variable, title=f"{target_variable} Over Time")
            st.plotly_chart(fig, use_container_width=True)
            
            # Store the required data in session state so Next button will be enabled
            st.session_state.df = df
            st.session_state.target_variable = target_variable
            st.session_state.freq = freq
            
            return df, target_variable, freq
            
        except Exception as e:
            st.error(f"Error processing data with frequency {freq}: {str(e)}")
            return None, None, None
            
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
        st.error("Please ensure your CSV file is properly formatted with a date column and numeric data for forecasting.")
        return None, None, None

def render_step2_model_configuration():
    """
    Render step 2: Model configuration
    
    Returns:
        tuple: (model_choice, train_size, model_params) - model selection and parameters
    """
    st.header("Step 2: Configure Model")
    
    if not st.session_state.get('data_uploaded', False):
        st.warning("Please complete Step 1 before configuring the model.")
        return None, None, {}
    
    # Model selection
    model_choice = st.selectbox("Select a forecasting model", 
                               ["ETS", "ARIMA", "RandomForest", "XGBoost"],
                               help="Choose the type of forecasting model to use")
    
    # Train/test split configuration
    st.subheader("Training Data Configuration")
    train_size = st.slider("Training data percentage", 
                          min_value=50, max_value=95, value=80, step=5,
                          help="Percentage of data to use for training. The rest will be used for validation.") / 100
    
    # Model parameters
    st.subheader("Model Parameters")
    model_params = render_model_params_ui(model_choice)
    
    # Automatically save configuration when proceeding to Step 3
    st.session_state.model_configured = True
    st.session_state.model_choice = model_choice
    st.session_state.train_size = train_size
    st.session_state.model_params = model_params
    
    return model_choice, train_size, model_params

def render_step3_results():
    """
    Render step 3: View forecasting results
    """
    st.header("Step 3: Forecast Results")
    
    if not st.session_state.get('model_configured', False):
        st.warning("Please complete Steps 1 and 2 before viewing results.")
        return
    
    # Get saved values from previous steps
    df = st.session_state.get('df')
    target_variable = st.session_state.get('target_variable')
    model_choice = st.session_state.get('model_choice')
    train_size = st.session_state.get('train_size')
    model_params = st.session_state.get('model_params')
    
    # Forecast horizon configuration
    fh = st.number_input("Forecast horizon (number of periods to forecast)", 
                        min_value=1, value=10, step=1,
                        help="Number of time periods to forecast into the future")
    
    run_forecast_button = st.button("Run Forecast")
    
    if run_forecast_button:
        # Show a spinner while calculating
        with st.spinner('Running forecast...'):
            try:
                from models import run_forecast
                from preprocess import manual_train_test_split
                from metrics import calculate_forecast_metrics, display_metrics, calculate_naive_baseline
                
                # Get the target series
                y = df[target_variable]
                
                # Split into train/test
                y_train, y_test = manual_train_test_split(y, train_size)
                
                # Run the forecast
                forecaster, y_pred, y_forecast = run_forecast(
                    y_train, y_test, model_choice, fh, **model_params
                )
                
                # Store the results for future reference
                st.session_state.forecaster = forecaster
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.y_forecast = y_forecast
                
                # Plot the results
                st.subheader("Forecast Results")
                
                # Combine all series for date range selection
                all_data = pd.concat([
                    pd.Series(y_train.values, index=y_train.index.to_timestamp(), name='Train'),
                    pd.Series(y_test.values, index=y_test.index.to_timestamp(), name='Test'),
                    pd.Series(y_pred.values, index=y_pred.index.to_timestamp(), name='Predictions'),
                    pd.Series(y_forecast.values, index=y_forecast.index.to_timestamp(), name='Forecast')
                ])
                
                # Date range selection for forecast plot
                min_date = all_data.index.min()
                max_date = all_data.index.max()
                
                st.subheader("Select Date Range for Plot")
                date_range = st.date_input(
                    "Plot date range",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
                
                # Handle both single date and date range selection
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = date_range
                    end_date = date_range
                
                # Convert to timestamps for filtering
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                
                # Filter data based on selected date range
                filtered_train = y_train[(y_train.index.to_timestamp() >= start_ts) & 
                                       (y_train.index.to_timestamp() <= end_ts)]
                filtered_test = y_test[(y_test.index.to_timestamp() >= start_ts) & 
                                     (y_test.index.to_timestamp() <= end_ts)]
                filtered_pred = y_pred[(y_pred.index.to_timestamp() >= start_ts) & 
                                     (y_pred.index.to_timestamp() <= end_ts)]
                filtered_forecast = y_forecast[(y_forecast.index.to_timestamp() >= start_ts) & 
                                            (y_forecast.index.to_timestamp() <= end_ts)]

                # Plot results with date range filtering
                import plotly.graph_objects as go
                fig = go.Figure()
                
                if not filtered_train.empty:
                    fig.add_trace(go.Scatter(
                        x=filtered_train.index.to_timestamp(), 
                        y=filtered_train.values, 
                        mode='lines', 
                        name='Train'
                    ))
                    
                if not filtered_test.empty:
                    fig.add_trace(go.Scatter(
                        x=filtered_test.index.to_timestamp(), 
                        y=filtered_test.values, 
                        mode='lines', 
                        name='Test'
                    ))
                    
                if not filtered_pred.empty:
                    fig.add_trace(go.Scatter(
                        x=filtered_pred.index.to_timestamp(), 
                        y=filtered_pred.values, 
                        mode='lines', 
                        name='Test Predictions'
                    ))
                    
                if not filtered_forecast.empty:
                    fig.add_trace(go.Scatter(
                        x=filtered_forecast.index.to_timestamp(), 
                        y=filtered_forecast.values, 
                        mode='lines', 
                        name='Forecast'
                    ))

                fig.update_layout(
                    title=f"{model_choice} Forecast for {target_variable}",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode="x unified",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Calculate and display metrics
                metrics = calculate_forecast_metrics(y_test, y_pred)
                st.subheader("Forecast Accuracy Metrics")
                st.table(display_metrics(metrics))

                # Calculate and display naïve baseline
                naive_metrics = calculate_naive_baseline(y_train, y_test)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Naïve Baseline Metrics")
                    st.metric("Naïve RMSE", round(naive_metrics["RMSE"], 2))
                    st.metric("Naïve MAE", round(naive_metrics["MAE"], 2))
                
                with col2:
                    st.subheader("Model Performance")
                    rmse_improvement = (naive_metrics["RMSE"] - metrics["RMSE"]) / naive_metrics["RMSE"] * 100
                    mae_improvement = (naive_metrics["MAE"] - metrics["MAE"]) / naive_metrics["MAE"] * 100
                    
                    st.metric("RMSE Improvement", f"{rmse_improvement:.2f}%", 
                            delta=f"{naive_metrics['RMSE'] - metrics['RMSE']:.2f}", 
                            delta_color="normal")
                    st.metric("MAE Improvement", f"{mae_improvement:.2f}%", 
                            delta=f"{naive_metrics["MAE"] - metrics["MAE"]:.2f}", 
                            delta_color="normal")
                
                # Detailed forecast results
                render_forecast_results_ui(forecaster, y_pred, y_forecast)
            
            except Exception as e:
                st.error(f"An error occurred during forecasting: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

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

def render_forecast_results_ui(forecaster, y_pred, y_forecast):
    """
    Render UI components for displaying forecast results
    
    Args:
        forecaster: Fitted forecaster model
        y_pred (pd.Series): Predictions on test data
        y_forecast (pd.Series): Future forecasts
    """
    from metrics import prepare_forecast_download
    
    with st.expander("View Test Set Predictions"):
        st.write(y_pred)

    with st.expander("View Future Forecast Values"):
        st.write(y_forecast)
    
    # Display model summary if available
    if hasattr(forecaster, 'summary'):
        with st.expander("Model Summary"):
            st.text(forecaster.summary())
            
    # Add download button for forecast results
    csv_data = prepare_forecast_download(y_forecast)
    st.download_button(
        label="Download Forecast CSV",
        data=csv_data,
        file_name="forecast_results.csv",
        mime="text/csv"
    )
