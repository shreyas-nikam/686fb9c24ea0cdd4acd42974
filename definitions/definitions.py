import pandas as pd
import numpy as np

def generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples):
    """Generates a synthetic Moving Average time series."""

    if order not in [1, 2]:
        raise ValueError("Order must be 1 or 2.")

    if order == 1 and len(theta_coeffs) != 1:
        raise ValueError("Incorrect number of theta coefficients for order 1.")
    elif order == 2 and len(theta_coeffs) != 2:
        raise ValueError("Incorrect number of theta coefficients for order 2.")

    # Generate white noise error terms
    epsilon = np.random.normal(0, epsilon_std_dev, num_samples)
    
    # Generate MA time series
    ma_data = np.zeros(num_samples)
    if order == 1:
        for t in range(1, num_samples):
            ma_data[t] = epsilon[t] + theta_coeffs[0] * epsilon[t-1]
    elif order == 2:
        for t in range(2, num_samples):
            ma_data[t] = epsilon[t] + theta_coeffs[0] * epsilon[t-1] + theta_coeffs[1] * epsilon[t-2]
    else:
        raise ValueError("Order must be 1 or 2.")

    # Create pandas Series with DatetimeIndex
    index = pd.date_range(start='2023-01-01', periods=num_samples, freq='M')
    ma_series = pd.Series(ma_data, index=index)
    
    return ma_series

import pandas as pd
import numpy as np
from scipy import stats

def calculate_acf(data, lags):
    """Computes ACF for a time series."""

    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series.")

    n = len(data)
    acf_values = np.full(lags + 1, np.nan)  # Initialize with NaN
    conf_int = np.full((2, lags + 1), np.nan)

    if n > 0:
        mean = np.nanmean(data)
        variance = np.nanvar(data)

        for lag in range(lags + 1):
            if lag == 0:
                acf_values[lag] = 1.0
            else:
                covariance = np.nanmean((data[:n - lag] - mean) * (data[lag:] - mean))
                acf_values[lag] = covariance / variance if variance > 0 else np.nan

        # Confidence interval calculation
        critical_value = stats.norm.ppf(0.975)  # For 95% confidence interval
        conf_interval = critical_value / np.sqrt(n)

        conf_int[0] = acf_values + conf_interval # Upper bound
        conf_int[1] = acf_values - conf_interval # Lower bound
            
    return acf_values, conf_int

import pandas as pd
import plotly.graph_objects as go

def plot_time_series(data):
    """Creates an interactive line plot of the time series data using Plotly.

    Args:
        data (pandas.Series): The time series data to plot.

    Returns:
        A Plotly `Figure` object.
    """
    if not pd.api.types.is_numeric_dtype(data):
        raise TypeError("Data must be numeric")

    fig = go.Figure(data=[go.Scatter(x=data.index, y=data.values)])
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="MA Value",
        template="plotly_white"
    )
    return fig

import plotly.graph_objects as go
import numpy as np

def plot_acf(acf_values, conf_int, title):
    """Generates an interactive correlogram (ACF plot) using Plotly.
    Args:
        acf_values (numpy.array): The autocorrelation coefficients.
        conf_int (numpy.array): The confidence interval bounds.
        title (str): Title for the ACF plot.
    Returns:
        A Plotly `Figure` object.
    """
    lags = len(acf_values)
    fig = go.Figure()

    # Add ACF bars
    fig.add_trace(go.Bar(x=np.arange(lags), y=acf_values, name='ACF'))

    # Add confidence interval
    if len(conf_int) > 0 and len(acf_values) > 0:
        upper_bound = conf_int[:, 1]
        lower_bound = conf_int[:, 0]

        fig.add_trace(go.Scatter(
            x=np.arange(lags),
            y=upper_bound,
            mode='lines',
            marker=dict(color='rgba(0, 0, 0, 0)'),
            showlegend=False,
            name='Upper Bound'
        ))

        fig.add_trace(go.Scatter(
            x=np.arange(lags),
            y=lower_bound,
            mode='lines',
            marker=dict(color='rgba(0, 0, 0, 0)'),
            fill='tonexty',
            fillcolor='rgba(0, 0, 0, 0.1)',
            showlegend=False,
            name='Lower Bound'
        ))

    # Add zero line
    fig.add_trace(go.Scatter(x=np.arange(lags), y=[0] * lags, mode='lines',
                             line=dict(color='red', width=1),
                             name='Zero Line'))

    # Update layout
    fig.update_layout(title=title,
                      xaxis_title='Lag',
                      yaxis_title='Autocorrelation',
                      xaxis=dict(tickmode='linear', tick0=0, dtick=1))

    return fig

import pandas as pd
import numpy as np
from IPython.display import display
# Assuming these functions are defined elsewhere and imported
from typing import List, Tuple
def generate_ma_data(order: int, theta_coeffs: List[float], epsilon_std_dev: float, num_samples: int = 24) -> pd.Series:
    """Generates MA data.  This is a stub."""
    pass

def calculate_acf(data: pd.Series, num_lags: int = 10) -> Tuple[List[float], List[float]]:
     """Calculates ACF. This is a stub."""
     pass

def plot_time_series(data: pd.Series, title: str):
     """Plots time series data.  This is a stub."""
     pass

def plot_acf(acf_values: List[float], confidence_intervals: List[float]):
     """Plots ACF.  This is a stub."""
     pass

def update_plots(ma_order: int, theta_1: float, theta_2: float, epsilon_std_dev: float):
    """Orchestrates data generation and plotting for MA models."""

    theta_coeffs = []
    if ma_order >= 1:
        theta_coeffs.append(theta_1)
    if ma_order >= 2:
        theta_coeffs.append(theta_2)

    # Generate MA data
    ma_data = generate_ma_data(order=ma_order, theta_coeffs=theta_coeffs, epsilon_std_dev=epsilon_std_dev, num_samples=24)

    # Calculate ACF
    acf_values, confidence_intervals = calculate_acf(ma_data)

    # Plot time series
    time_series_plot = plot_time_series(ma_data, title="Moving Average Time Series")
    display(time_series_plot)

    # Plot ACF
    acf_plot = plot_acf(acf_values, confidence_intervals)
    display(acf_plot)