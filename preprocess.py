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

def load_and_preprocess_data(uploaded_file, freq):
    """
    Load and preprocess time series data from a CSV file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        freq (str): Time frequency ('D', 'W', 'M', 'Q', 'Y')
    
    Returns:
        pd.DataFrame: Preprocessed dataframe with date index
    """
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Convert the index to datetime and then to PeriodIndex
        df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df.set_index('date')
        df = df.sort_index()  # Ensure the index is sorted
        df.index = df.index.to_period(freq)
        
        # Remove any rows with NaT in the index
        df = df.loc[df.index.notnull()]
        
        return df
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