import pandas as pd
import numpy as np
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def create_features(y, lags=3, rolling_windows=[3, 7, 14]):
    """
    Create time series features from datetime index
    
    Args:
        y (pd.Series): Time series data
        lags (int): Number of lag features to create
        rolling_windows (list): List of window sizes for rolling statistics
        
    Returns:
        tuple: (X, y_filtered) - features and corresponding target values
    """
    df = pd.DataFrame({'y': y})
    
    # Date-based features
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    if hasattr(df.index, 'dayofweek'):
        df['dayofweek'] = df.index.dayofweek
    df['year'] = df.index.year
    
    # Lag features
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    
    # Rolling window features
    for window in rolling_windows:
        if len(df) > window:  # Only create if we have enough data
            df[f'rolling_mean_{window}'] = df['y'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['y'].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['y'].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['y'].rolling(window=window).max()
    
    # Drop NA values from feature creation
    df = df.dropna()
    
    # Target is still 'y', features are everything else
    X = df.drop('y', axis=1)
    y_filtered = df['y']
    
    return X, y_filtered

def recursive_forecast_ml(model, X_test_start, future_steps, freq):
    """
    Generate forecasts using recursive strategy for ML models
    
    Args:
        model: Fitted ML model
        X_test_start (pd.DataFrame): Starting feature set for forecasting
        future_steps (int): Number of steps to forecast
        freq (str): Frequency of the time series
        
    Returns:
        pd.Series: Forecasted values
    """
    last_idx = X_test_start.index[-1]
    future_idx = pd.period_range(start=last_idx + 1, periods=future_steps, freq=freq)
    forecasts = []
    
    # Make a copy to avoid modifying the original
    X_current = X_test_start.copy()
    
    for i in range(future_steps):
        # Predict the next value
        next_pred = model.predict(X_current.iloc[[-1]])[0]
        forecasts.append(next_pred)
        
        # Create new row for next prediction
        new_row = X_current.iloc[[-1]].copy()
        new_idx = future_idx[i]
        new_row.index = [new_idx]
        
        # Update date features
        new_row['month'] = new_idx.month
        new_row['quarter'] = new_idx.quarter
        if hasattr(new_idx, 'dayofweek'):
            new_row['dayofweek'] = new_idx.dayofweek
        new_row['year'] = new_idx.year
        
        # Update lag features
        for lag in range(1, len([col for col in X_current.columns if col.startswith('lag_')]) + 1):
            if f'lag_{lag}' in new_row.columns:
                if lag == 1:
                    new_row[f'lag_{lag}'] = next_pred
                else:
                    new_row[f'lag_{lag}'] = X_current.iloc[-1][f'lag_{lag-1}']
        
        # Update rolling features if they exist
        for window in [3, 7, 14]:
            if f'rolling_mean_{window}' in new_row.columns:
                # Simple approximation for rolling mean
                new_row[f'rolling_mean_{window}'] = (X_current.iloc[-1][f'rolling_mean_{window}'] * (window-1) + next_pred) / window
                
                # For other features, maintain the last value as an approximation
                # In a more sophisticated implementation, these would be properly updated
        
        # Append new row to X_current
        X_current = pd.concat([X_current, new_row])
    
    return pd.Series(forecasts, index=future_idx)

def run_forecast(y_train, y_test, model, fh, **kwargs):
    """
    Run forecast for selected model and data
    
    Args:
        y_train (pd.Series): Training data
        y_test (pd.Series): Test data
        model (str): Model name ('ETS', 'ARIMA', 'RandomForest', 'XGBoost')
        fh (int): Forecast horizon (number of periods)
        **kwargs: Model-specific parameters
        
    Returns:
        tuple: (forecaster, y_pred, y_forecast) - fitted model, predictions, and forecast
    """
    if model in ['RandomForest', 'XGBoost']:
        # Extract lag parameter if provided, otherwise default to 3
        lags = kwargs.pop('lags', 3)
        
        # Create features for train set with user-specified lags
        X_train, y_train_filtered = create_features(y_train, lags=lags)
        
        # Initialize model
        if model == 'RandomForest':
            forecaster = RandomForestRegressor(**kwargs)
        else:  # XGBoost
            forecaster = XGBRegressor(**kwargs)
            
        # Fit model
        forecaster.fit(X_train, y_train_filtered)
        
        # Create features for test set (one step at a time for proper rolling features)
        y_full = pd.concat([y_train, y_test])
        preds = []
        
        # For each test point
        for i in range(len(y_train), len(y_full)):
            # Get data up to this point
            current_history = y_full.iloc[:i]
            X_test_point, _ = create_features(current_history, lags=lags)
            if X_test_point.empty:
                # If we don't have enough history data for feature creation
                preds.append(y_train.iloc[-1])  # Use last training value
                continue
                
            X_test_last_row = X_test_point.iloc[-1:] 
            
            # Predict and store
            pred = forecaster.predict(X_test_last_row)[0]
            preds.append(pred)
        
        y_pred = pd.Series(preds, index=y_test.index)
        
        # Future forecast using recursive forecasting
        # Get full history including test set
        full_history = pd.concat([y_train, y_test])
        X_last, _ = create_features(full_history, lags=lags)
        
        if X_last.empty:
            # Fallback if feature creation doesn't work
            y_forecast = pd.Series([y_test.iloc[-1]] * fh, 
                        index=pd.period_range(start=y_test.index[-1] + 1, periods=fh, freq=y_train.index.freq))
        else:
            # Use recursive forecasting for future predictions
            y_forecast = recursive_forecast_ml(forecaster, X_last, fh, y_train.index.freq)
    else:
        # Code for ETS and ARIMA models with duplicate handling
        if model == 'ETS':
            forecaster = AutoETS(**kwargs)
        elif model == 'ARIMA':
            forecaster = AutoARIMA(**kwargs)
        
        forecaster.fit(y_train)
        
        # Handle potential duplicate indices in test data
        test_indices = y_test.index.drop_duplicates()
        y_pred = forecaster.predict(fh=ForecastingHorizon(test_indices, is_relative=False))
        
        # Forecast future values
        last_date = y_test.index[-1]
        future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq).drop_duplicates()
        future_horizon = ForecastingHorizon(future_dates, is_relative=False)
        y_forecast = forecaster.predict(fh=future_horizon)
        
        # Add a summary method to the forecaster objects if they don't already have one
        if not hasattr(forecaster, 'summary'):
            forecaster.summary = lambda: str(forecaster)
    
    return forecaster, y_pred, y_forecast

def get_model_params(model_choice):
    """
    Return default parameters for each model type
    
    Args:
        model_choice (str): Model name
        
    Returns:
        dict: Default parameters for the selected model
    """
    if model_choice == "ETS":
        return {
            "error": "add",
            "trend": "add",
            "seasonal": "add",
            "damped_trend": False,
            "sp": 1
        }
    elif model_choice == "ARIMA":
        return {
            "start_p": 0,
            "max_p": 5,
            "start_q": 0,
            "max_q": 5,
            "d": 1,
            "seasonal": True,
            "start_P": 0,
            "max_P": 2,
            "start_Q": 0,
            "max_Q": 2,
            "D": 1,
            "sp": 12
        }
    elif model_choice == "RandomForest":
        return {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "lags": 3  # Default lag value
        }
    elif model_choice == "XGBoost":
        return {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "lags": 3  # Default lag value
        }
    else:
        return {}