import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_forecast_metrics(y_true, y_pred):
    """
    Calculate common forecast accuracy metrics
    
    Args:
        y_true (pd.Series): Actual values
        y_pred (pd.Series): Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Mean Squared Error
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    
    # Mean Absolute Error
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error (with handling for zero values)
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    # Avoid division by zero
    nonzero_indices = y_true_array != 0
    if np.any(nonzero_indices):
        mape = np.mean(np.abs((y_true_array[nonzero_indices] - y_pred_array[nonzero_indices]) / 
                              y_true_array[nonzero_indices])) * 100
        metrics['MAPE'] = mape
    else:
        metrics['MAPE'] = np.nan
    
    # R-squared
    metrics['R2'] = r2_score(y_true, y_pred)
    
    return metrics

def calculate_naive_baseline(y_train, y_test):
    """
    Calculate naïve baseline metrics (using last known value for all predictions)
    
    Args:
        y_train (pd.Series): Training data
        y_test (pd.Series): Test data
        
    Returns:
        dict: Dictionary of naïve metrics
    """
    # Generate naïve predictions (last training value repeated)
    naive_pred = pd.Series(y_train.iloc[-1], index=y_test.index)
    
    # Calculate metrics
    naive_mse = mean_squared_error(y_test, naive_pred)
    naive_rmse = np.sqrt(naive_mse)
    naive_mae = mean_absolute_error(y_test, naive_pred)
    
    return {
        "MSE": naive_mse,
        "RMSE": naive_rmse, 
        "MAE": naive_mae
    }

def display_metrics(metrics):
    """
    Format metrics for display
    
    Args:
        metrics (dict): Dictionary of metrics
        
    Returns:
        pd.DataFrame: Formatted metrics for display
    """
    df_metrics = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': [f"{value:.4f}" for value in metrics.values()]
    })
    return df_metrics

def prepare_forecast_download(y_forecast):
    """
    Prepare forecast data for download as CSV
    
    Args:
        y_forecast (pd.Series): Forecast data
        
    Returns:
        bytes: CSV data encoded as bytes for download
    """
    csv = y_forecast.to_frame(name="Forecast").reset_index()
    csv.columns = ["Date", "Forecast"]
    csv_data = csv.to_csv(index=False).encode("utf-8")
    return csv_data
