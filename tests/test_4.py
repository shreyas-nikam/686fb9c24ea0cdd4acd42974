import pytest
from definition_a583f4c407c842d284e2c233bfee2771 import update_plots
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_acf
import plotly.graph_objects as go
from ipywidgets import interactive, Dropdown, FloatSlider
from IPython.display import display

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

@pytest.fixture
def mock_functions(monkeypatch):
    def mock_generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples):
        return pd.Series([1,2,3], index=pd.date_range(start='2023-01-01', periods=3, freq='M'))
    
    def mock_calculate_acf(data, lags):
        return np.array([0.1, 0.2, 0.3]), np.array([[0,0,0],[0.2,0.3,0.4]])

    def mock_plot_time_series(data):
        return go.Figure()

    def mock_plot_acf(acf_values, conf_int, title):
        return go.Figure()

    monkeypatch.setattr('definition_a583f4c407c842d284e2c233bfee2771.generate_ma_data', mock_generate_ma_data)
    monkeypatch.setattr('definition_a583f4c407c842d284e2c233bfee2771.calculate_acf', mock_calculate_acf)
    monkeypatch.setattr('definition_a583f4c407c842d284e2c233bfee2771.plot_time_series', mock_plot_time_series)
    monkeypatch.setattr('definition_a583f4c407c842d284e2c233bfee2771.plot_acf', mock_plot_acf)


def test_update_plots_ma1(mock_functions, monkeypatch):
    """Test MA(1) with valid parameters."""
    plots_displayed = False
    def mock_display(fig):
        nonlocal plots_displayed
        plots_displayed = True
    monkeypatch.setattr("IPython.display.display", mock_display)

    update_plots(ma_order=1, theta_1=0.5, theta_2=0.2, epsilon_std_dev=1.0)
    assert plots_displayed

def test_update_plots_ma2(mock_functions, monkeypatch):
    """Test MA(2) with valid parameters."""
    plots_displayed = False
    def mock_display(fig):
        nonlocal plots_displayed
        plots_displayed = True
    monkeypatch.setattr("IPython.display.display", mock_display)
    update_plots(ma_order=2, theta_1=0.5, theta_2=0.2, epsilon_std_dev=1.0)
    assert plots_displayed

def test_update_plots_invalid_epsilon(mock_functions, monkeypatch):
    """Test with invalid epsilon standard deviation (negative)."""
    plots_displayed = False
    def mock_display(fig):
        nonlocal plots_displayed
        plots_displayed = True
    monkeypatch.setattr("IPython.display.display", mock_display)
    update_plots(ma_order=1, theta_1=0.5, theta_2=0.2, epsilon_std_dev=-1.0) # this will still plot but there should be a handling of exception if needed
    assert plots_displayed