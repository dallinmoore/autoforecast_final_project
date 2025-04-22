# Author: Prof. Pedram Jahangiry
# Date: 2024-10-10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def manual_train_test_split(y, train_size):
    """
    Split a time series into train and test sets.
    
    Args:
        y (pd.Series): Time series data
        train_size (float): Proportion of data to use for training (0.0-1.0)
    
    Returns:
        tuple: (y_train, y_test) - training and testing data
    """
    split_point = int(len(y) * train_size)
    return y[:split_point], y[split_point:]

def load_and_preprocess_data(uploaded_file, freq=None):
    """
    Load and preprocess time series data from a CSV file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        freq (str): Time frequency ('h', 'D', 'W', 'M', 'Q', 'Y')
    
    Returns:
        tuple: (df, freq, datetime_column) - preprocessed dataframe with date index, the frequency, and datetime column name
    """
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Find datetime columns automatically
        datetime_candidates = [
            col for col in df.columns
            if (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]))
            and pd.to_datetime(df[col], errors='coerce').notna().sum() > 0
        ]
        
        if not datetime_candidates:
            raise ValueError("No valid datetime-like columns found in the data.")
            
        # Use the first datetime column found
        datetime_column = datetime_candidates[0]
        
        # Convert to datetime and set as index
        df['datetime'] = pd.to_datetime(df[datetime_column], errors='coerce')
        df = df.dropna(subset=['datetime']).drop_duplicates(subset='datetime').sort_values('datetime')
        df = df.set_index('datetime')
        
        # Infer frequency if not provided
        if freq is None:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                # Standardize frequency - convert variations to standard forms
                if inferred_freq == 'MS':
                    freq = 'M'
                elif inferred_freq == 'QS':
                    freq = 'Q'
                elif inferred_freq == 'YS':
                    freq = 'Y'
                else:
                    freq = inferred_freq
            else:
                freq = 'D'  # default to daily if can't infer
        
        # Convert index to PeriodIndex with the given frequency
        df.index = df.index.to_period(freq)
        
        # Remove any rows with NaT in the index
        df = df.loc[df.index.notnull()]
        
        return df, freq, datetime_column
        
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")

def plot_time_series(y_train, y_test, y_pred, y_forecast, title):
    """
    Plot time series data including training, test, predictions, and forecast.
    
    Args:
        y_train (pd.Series): Training data
        y_test (pd.Series): Test data
        y_pred (pd.Series): Predictions on test data
        y_forecast (pd.Series): Future forecast
        title (str): Plot title
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
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

def plot_original_series(df, target_variable):
    """
    Plot the original time series data.
    
    Args:
        df (pd.DataFrame): Dataframe containing time series data
        target_variable (str): Column name of the target variable
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index.to_timestamp(), df[target_variable])
    plt.title(f"{target_variable} Time Series")
    plt.xlabel("Date")
    plt.ylabel("Value")
    return fig