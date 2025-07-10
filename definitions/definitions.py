import pandas as pd
import numpy as np

def generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples):
    """Generates synthetic MA time series data."""

    if num_samples <= 0:
        raise ValueError("Number of samples must be positive.")
    if order not in [1, 2]:
        raise ValueError("Order must be 1 or 2.")
    if len(theta_coeffs) != order:
        raise ValueError("Length of theta_coeffs must match the order.")

    # Generate white noise error terms
    epsilon_values = np.random.normal(loc=0, scale=epsilon_std_dev, size=num_samples + order)

    # Generate MA series
    ma_values = np.zeros(num_samples + order)
    for t in range(order, num_samples + order):
        ma_values[t] = epsilon_values[t]
        for i in range(order):
            ma_values[t] += theta_coeffs[i] * epsilon_values[t - (i + 1)]

    # Convert to pandas Series and return
    return pd.Series(ma_values[order:])

import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_acf(data, lags):
    """Computes ACF for a time series."""
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series.")
    if data.empty:
        raise Exception("Input data cannot be empty.")
    if not isinstance(lags, int):
        raise TypeError("Lags must be an integer.")
    if lags < 0:
        raise ValueError("Lags must be a non-negative integer.")

    acf_values = np.zeros(lags + 1)
    n = len(data)
    mean = data.mean()
    variance = data.var()

    for lag in range(lags + 1):
        if lag == 0:
            acf_values[lag] = 1.0
        else:
            covariance = ((data[:n - lag] - mean) * (data[lag:] - mean)).sum() / n
            acf_values[lag] = covariance / variance

    conf_int = np.zeros((lags + 1, 2))
    z = norm.ppf(0.975)  # ~95% confidence interval
    for lag in range(lags + 1):
        conf_int[lag, 0] = -z / np.sqrt(n)
        conf_int[lag, 1] = z / np.sqrt(n)

    return acf_values, conf_int

import plotly.graph_objects as go
import pandas as pd

def plot_time_series(data):
    """Creates an interactive line plot of the time series data using Plotly.
    Args:
        data (pandas.Series): The time series data to plot.
    Output:
        A Plotly `Figure` object.
    """
    fig = go.Figure(data=[go.Scatter(x=data.index, y=data.values)])
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Value"
    )
    return fig

import plotly.graph_objects as go
import numpy as np

def plot_acf(acf_values, conf_int, title):
    """Generates an interactive correlogram (ACF plot) using Plotly."""

    if len(acf_values) == 0:
        raise IndexError("ACF values array is empty.")

    x = np.arange(len(acf_values))
    
    fig = go.Figure()

    # Add the ACF values as a bar plot
    fig.add_trace(go.Bar(x=x, y=acf_values, name='ACF'))

    # Add the confidence interval as a shaded region
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([conf_int[:, 1], conf_int[::-1, 0]]),
        fill='tozeroy',
        fillcolor='rgba(0, 0, 150, 0.2)',
        line=dict(color='rgba(0, 0, 0, 0)'),
        hoverinfo='skip',
        showlegend=False
    ))

    # Customize the layout
    fig.update_layout(
        title=title,
        xaxis_title='Lag',
        yaxis_title='Autocorrelation',
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),  # Integer x-axis
        yaxis=dict(range=[-1.05, 1.05]),
        shapes=[dict(
            type='line',
            x0=0,
            x1=len(acf_values) -1,
            y0=0,
            y1=0,
            line=dict(color='red', width=2)
        )],
        template="plotly_white"
    )

    return fig

def update_plots(ma_order, theta_1, theta_2, epsilon_std_dev):
    """Orchestrates data generation and plotting for MA models."""
    num_samples = 100
    lags = 20

    if ma_order == 1:
        theta_coeffs = [theta_1]
    elif ma_order == 2:
        theta_coeffs = [theta_1, theta_2]
    else:
        raise ValueError("MA order must be 1 or 2.")

    data = generate_ma_data(ma_order, theta_coeffs, epsilon_std_dev, num_samples)
    acf_values, conf_int = calculate_acf(data, lags)

    time_series_fig = plot_time_series(data)
    acf_fig = plot_acf(acf_values, conf_int, title='Autocorrelation Function')

    display(time_series_fig)
    display(acf_fig)


def generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples):
    ar = np.array([1])
    ma = np.concatenate(([1], theta_coeffs))
    epsilon = np.random.normal(0, epsilon_std_dev, num_samples)
    simulated_data = arma_generate_sample(ar, ma, num_samples, disturbances=epsilon)
    index = pd.date_range(start='2023-01-01', periods=num_samples, freq='M')
    return pd.Series(simulated_data, index=index)


def calculate_acf(data, lags):
    acf_values = plot_acf(data, lags=lags, auto_ylims=True, title=None)
    acf_result = acf_values.axes[0].lines[0].get_ydata()
    conf_int = acf_values.axes[0].lines[-1].get_ydata()
    return acf_result, conf_int


def plot_time_series(data):
    fig = go.Figure(data=[go.Scatter(x=data.index, y=data.values)])
    fig.update_layout(title='Generated MA Time Series', xaxis_title='Time', yaxis_title='MA Value')
    return fig


def plot_acf(acf_values, conf_int, title):
    lags = np.arange(len(acf_values))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags, y=acf_values, name='ACF'))
    fig.add_trace(go.Scatter(x=lags, y=conf_int[1], mode='lines', name='Upper CI', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=lags, y=conf_int[0], mode='lines', name='Lower CI', line=dict(dash='dash')))
    fig.update_layout(title=title, xaxis_title='Lag', yaxis_title='Autocorrelation')
    return fig