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
    .right-button {
        float: right;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("---")
    
    # Use a container for better alignment
    nav_container = st.container()
    
    # Create a row with appropriate column widths for navigation buttons
    prev_col, spacer, next_col = nav_container.columns([1, 10, 1])
    
    with prev_col:
        if st.session_state.current_step > 1:
            if st.button("Previous"):
                st.session_state.current_step -= 1
                st.rerun()
    
    with next_col:
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
                
            next_button = st.button("Next", disabled=not can_proceed, key="next_button")
            if next_button:
                # If moving from step 1 to 2, automatically save the data choices
                if st.session_state.current_step == 1:
                    st.session_state.data_uploaded = True
                # If moving from step 2 to 3, automatically save model configuration
                elif st.session_state.current_step == 2:
                    st.session_state.model_configured = True
                
                st.session_state.current_step += 1
                st.rerun()
    
    # For truly right-aligned button using custom HTML
    if st.session_state.current_step < 3:
        st.markdown(
            """
            <style>
            div[data-testid="stButton"][data-baseweb="button"] {
                float: right;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    
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
                            delta=f"{naive_metrics['MAE'] - metrics['MAE']:.2f}", 
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
        # Get time series data from session state
        if 'df' in st.session_state and 'target_variable' in st.session_state:
            df = st.session_state.df
            target_variable = st.session_state.target_variable
            ts = df[target_variable]
            
            # Check for negativity in the series
            has_negative = (ts < 0).any()
            
            # Auto-detect trend
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Try to determine if trend is present
            try:
                # Use statsmodels to decompose the time series
                freq = st.session_state.freq
                period = 1
                
                # Try to determine seasonal period based on frequency
                if freq == 'h':
                    period = 24  # 24 hours in a day
                elif freq == 'D':
                    period = 7   # 7 days in a week
                elif freq == 'W':
                    period = 52  # 52 weeks in a year
                elif freq == 'M':
                    period = 12  # 12 months in a year
                elif freq == 'Q':
                    period = 4   # 4 quarters in a year
                
                # Only attempt decomposition if we have enough data points
                has_trend = False
                has_seasonality = False
                
                if len(ts) >= 2 * period and period > 1:
                    # Convert to datetime index to use seasonal_decompose
                    temp_ts = ts.copy()
                    if not isinstance(temp_ts.index, pd.DatetimeIndex):
                        temp_ts.index = temp_ts.index.to_timestamp()
                    
                    try:
                        decomposition = seasonal_decompose(temp_ts, model='additive', period=period)
                        trend_component = decomposition.trend.dropna()
                        
                        # Check if trend is significant
                        if len(trend_component) > 0:
                            trend_range = trend_component.max() - trend_component.min()
                            data_range = ts.max() - ts.min()
                            has_trend = trend_range > 0.1 * data_range
                            
                        # Check if seasonality is significant
                        seasonal_component = decomposition.seasonal.dropna()
                        if len(seasonal_component) > 0:
                            seasonal_range = seasonal_component.max() - seasonal_component.min()
                            has_seasonality = seasonal_range > 0.05 * data_range
                    except:
                        # If decomposition fails, use simpler heuristics
                        has_trend = True  # Assume trend by default
                        has_seasonality = period > 1  # Assume seasonality if period > 1
                else:
                    # Simple trend detection for short series
                    if len(ts) > 5:
                        # Check if there's consistent increase/decrease
                        has_trend = abs(ts.iloc[-1] - ts.iloc[0]) > 0.1 * ts.std() * (len(ts) ** 0.5)
                    else:
                        has_trend = True  # Default to True for very short series
                    
                    has_seasonality = period > 1 and len(ts) >= period  # Simple seasonality check
            except:
                # Default values if auto-detection fails
                has_trend = True
                has_seasonality = period > 1
            
            # Display auto-detected information
            with st.expander("📊 View Automated Parameter Analysis Results"):
                info_cols = st.columns(1)
                with info_cols[0]:
                    if has_negative:
                        st.markdown("• Series contains negative values - additive error is recommended")
                    else:
                        st.markdown("• Series is non-negative - multiplicative error may be suitable")
                        
                    if has_trend:
                        st.markdown("• Trend detected in the data")
                    else:
                        st.markdown("• No significant trend detected")
                        
                    if has_seasonality:
                        st.markdown(f"• Seasonal pattern detected with period {period}")
                    else:
                        st.markdown("• No significant seasonality detected")
        
            # Make recommendations based on auto-detection
            error_default = "add" if has_negative else "mul"
            trend_default = "add" if has_trend else None
            seasonal_default = "add" if has_seasonality else None
            seasonal_period_default = period if has_seasonality else 1
            
            st.subheader("ETS Parameters")
            st.markdown("Based on your data characteristics, the following parameters are suggested:")
            
            error_options = ["add", "mul"]
            trend_options = ["add", "mul", None]
            seasonal_options = ["add", "mul", None]
            
            error = st.selectbox("Error type", error_options, 
                                index=error_options.index(error_default),
                                help="'add' for additive errors (suitable for any data), 'mul' for multiplicative errors (only for positive data)")
            
            trend = st.selectbox("Trend type", trend_options, 
                               index=trend_options.index(trend_default),
                               help="'add' for additive trend, 'mul' for multiplicative trend, None for no trend")
            
            seasonal = st.selectbox("Seasonal type", seasonal_options, 
                                  index=seasonal_options.index(seasonal_default),
                                  help="'add' for additive seasonality, 'mul' for multiplicative seasonality, None for no seasonality")
            
            damped_trend = st.checkbox("Damped trend", value=False,
                                     help="Damping reduces the impact of the trend component as the forecast horizon increases")
            
            seasonal_periods = st.number_input("Seasonal periods", min_value=1, value=seasonal_period_default,
                                             help="Number of time steps in one seasonal cycle")
        else:
            # If no data is available, use basic interface
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
        # Get time series data from session state
        if 'df' in st.session_state and 'target_variable' in st.session_state:
            df = st.session_state.df
            target_variable = st.session_state.target_variable
            ts = df[target_variable]
            freq = st.session_state.freq
            
            # Auto-detection for ARIMA parameters
            import pandas as pd
            import numpy as np
            from statsmodels.tsa.stattools import adfuller, acf, pacf
            
            # Detect stationarity using ADF test to suggest d value
            try:
                adf_result = adfuller(ts)
                is_stationary = adf_result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
                suggested_d = 0 if is_stationary else 1
            except:
                suggested_d = 1  # Default if test fails
            
            # Detect seasonality and suggest seasonal period
            seasonal_period = 1
            if freq == 'h':
                seasonal_period = 24  # 24 hours in a day
            elif freq == 'D':
                seasonal_period = 7   # 7 days in a week
            elif freq == 'W':
                seasonal_period = 52  # 52 weeks in a year
            elif freq == 'M':
                seasonal_period = 12  # 12 months in a year
            elif freq == 'Q':
                seasonal_period = 4   # 4 quarters in a year
            
            # Detect if seasonality exists
            has_seasonality = False
            try:
                if len(ts) >= 2 * seasonal_period:
                    # Calculate autocorrelation at seasonal lag
                    autocorr = acf(ts, nlags=seasonal_period)
                    has_seasonality = abs(autocorr[seasonal_period]) > 0.3  # Significant autocorrelation at seasonal lag
            except:
                has_seasonality = seasonal_period > 1  # Assume seasonality if period > 1
            
            # Suggest p and q based on ACF and PACF if enough data
            suggested_p = 1
            suggested_q = 1
            suggested_P = 1
            suggested_Q = 1
            suggested_D = 1 if has_seasonality else 0
            
            # Display auto-detection results in a compact info box
            with st.expander("📊 View Automated Parameter Analysis Results"):
                info_cols = st.columns(2)
                with info_cols[0]:
                    st.subheader("Non-Seasonal Analysis")
                    st.markdown(f"• **Stationarity**: {'Yes ✓' if is_stationary else 'No ✗'}")
                    st.markdown(f"• **Suggested d**: {suggested_d}")
                    st.markdown(f"• **Suggested p**: {suggested_p}")
                    st.markdown(f"• **Suggested q**: {suggested_q}")
                
                with info_cols[1]:
                    st.subheader("Seasonal Analysis")
                    st.markdown(f"• **Seasonality detected**: {'Yes ✓' if has_seasonality else 'No ✗'}")
                    st.markdown(f"• **Seasonal period**: {seasonal_period}")
                    if has_seasonality:
                        st.markdown(f"• **Suggested P**: {suggested_P}")
                        st.markdown(f"• **Suggested Q**: {suggested_Q}")
                        st.markdown(f"• **Suggested D**: {suggested_D}")
                        
            # Non-seasonal parameters
            st.subheader("ARIMA Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Autoregressive (AR)**")
                p = st.number_input(
                    "p value", 
                    min_value=0, 
                    max_value=5, 
                    value=suggested_p,
                    help="Number of lag observations (AR order). Higher values make the model consider more past observations."
                )
                
                d = st.number_input(
                    "d value (differencing)", 
                    min_value=0, 
                    max_value=2, 
                    value=suggested_d,
                    help="Differencing order needed to make the time series stationary. Usually 1 for most series, 0 if already stationary."
                )
                
            with col2:
                st.markdown("**Moving Average (MA)**")
                q = st.number_input(
                    "q value", 
                    min_value=0, 
                    max_value=5, 
                    value=suggested_q,
                    help="Size of the moving average window (MA order). Controls how many past forecast errors to use."
                )
                
            # Seasonal parameters
            st.subheader("Seasonality")
            use_seasonal = st.radio(
                "Include seasonal component?", 
                ["Yes", "No"],
                index=0 if has_seasonality else 1,
                help="Select 'Yes' if your data shows regular patterns that repeat at fixed intervals (like yearly seasons, weekly patterns)"
            )
            
            include_seasonal = use_seasonal == "Yes"
            
            # Initialize model parameters
            model_params = {
                "start_p": p,
                "max_p": p,
                "start_q": q,
                "max_q": q,
                "d": d,
                "seasonal": include_seasonal,
            }
            
            if include_seasonal:
                st.markdown("**Seasonal Parameters**")
                
                col1, col2 = st.columns(2)
                with col1:
                    P = st.number_input(
                        "P value (seasonal AR)", 
                        min_value=0, 
                        max_value=2, 
                        value=suggested_P,
                        help="Seasonal autoregressive order. Controls effects of observations from previous seasonal cycles."
                    )
                    
                    D = st.number_input(
                        "D value (seasonal differencing)", 
                        min_value=0, 
                        max_value=1, 
                        value=suggested_D,
                        help="Seasonal differencing order. Usually 1 if there's a repeating seasonal pattern."
                    )
                    
                with col2:
                    Q = st.number_input(
                        "Q value (seasonal MA)", 
                        min_value=0, 
                        max_value=2, 
                        value=suggested_Q,
                        help="Seasonal moving average order. Controls effects of errors from previous seasonal cycles."
                    )
                    
                    s = st.number_input(
                        "Seasonal period (s)", 
                        min_value=2, 
                        value=seasonal_period,
                        help="Number of time steps in each seasonal cycle. For example, 12 for monthly data with yearly seasonality."
                    )
                
                model_params.update({
                    "start_P": P,
                    "max_P": P,
                    "start_Q": Q,
                    "max_Q": Q,
                    "D": D,
                    "sp": s
                })
        else:
            # Basic interface if no data is available
            st.subheader("ARIMA Parameters")
            st.warning("Load data in Step 1 to enable parameter auto-detection")
            
            col1, col2 = st.columns(2)
            with col1:
                p = st.number_input("p value", min_value=0, value=1, 
                                   help="Autoregressive order")
                d = st.number_input("d value", min_value=0, value=1,
                                   help="Differencing order")
            
            with col2:
                q = st.number_input("q value", min_value=0, value=1,
                                   help="Moving average order")
            
            st.subheader("Seasonality")
            use_seasonal = st.radio("Include seasonal component?", ["Yes", "No"])
            include_seasonal = use_seasonal == "Yes"
            
            model_params = {
                "start_p": p,
                "max_p": p,
                "start_q": q,
                "max_q": q,
                "d": d,
                "seasonal": include_seasonal,
            }
            
            if include_seasonal:
                col1, col2 = st.columns(2)
                with col1:
                    P = st.number_input("P value", min_value=0, value=1,
                                       help="Seasonal autoregressive order")
                    D = st.number_input("D value", min_value=0, value=1,
                                       help="Seasonal differencing order")
                
                with col2:
                    Q = st.number_input("Q value", min_value=0, value=1,
                                       help="Seasonal moving average order")
                    s = st.number_input("Seasonal period", min_value=2, value=12,
                                       help="Number of time steps in seasonal cycle")
                
                model_params.update({
                    "start_P": P,
                    "max_P": P,
                    "start_Q": Q,
                    "max_Q": Q,
                    "D": D,
                    "sp": s
                })
        
        return model_params
    
    elif model_choice == "RandomForest" or model_choice == "XGBoost":
        # Common ML parameters section
        st.subheader("Time Series Parameters")
        lags = st.number_input("Number of lags", min_value=1, value=3,
                               help="Number of past time steps to use as features. Higher values capture longer-term patterns.")
        
        # Model-specific parameters
        st.subheader("Model Parameters")
        
        if model_choice == "RandomForest":
            n_estimators = st.number_input("Number of trees", min_value=10, value=100,
                                          help="Number of trees in the forest. More trees generally improve performance but increase computation time.")
            max_depth = st.number_input("Max depth", min_value=1, value=10,
                                       help="Maximum depth of each tree. Deeper trees can model more complex patterns but may overfit.")
            
            return {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": 42,
                "lags": lags  # Pass the lag parameter to the model
            }
        
        elif model_choice == "XGBoost":
            learning_rate = st.number_input("Learning rate", min_value=0.01, value=0.1,
                                          help="Step size shrinkage used to prevent overfitting. Lower values require more trees.")
            n_estimators = st.number_input("Number of trees", min_value=10, value=100,
                                          help="Number of gradient boosted trees. More trees generally improve performance.")
            max_depth = st.number_input("Max depth", min_value=1, value=10,
                                       help="Maximum depth of each tree. Deeper trees can model more complex patterns but may overfit.")
            
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
