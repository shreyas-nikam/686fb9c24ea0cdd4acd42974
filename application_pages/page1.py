
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf
from scipy.stats import norm

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
        raise ValueError("Order must be 1 or 2.") # This line is technically redundant due to the initial check.

    # Create pandas Series with DatetimeIndex
    index = pd.date_range(start='2023-01-01', periods=num_samples, freq='M')
    ma_series = pd.Series(ma_data, index=index)
    
    return ma_series

from statsmodels.tsa.stattools import acf
from scipy.stats import norm

def calculate_acf(series, nlags=20, alpha=0.05):
    """Calculates the Autocorrelation Function (ACF) and confidence intervals.

    Args:
        series (pd.Series): The time series data.
        nlags (int): The number of lags to calculate the ACF for.
        alpha (float): The significance level for the confidence intervals.

    Returns:
        tuple: A tuple containing:
            - acf_values (np.ndarray): The ACF values.
            - conf_int_lower (np.ndarray): The lower bound of the confidence interval.
            - conf_int_upper (np.ndarray): The upper bound of the confidence interval.
    """
    acf_values = acf(series, nlags=nlags, fft=True)
    
    # Calculate confidence intervals
    n = len(series)
    critical_value = norm.ppf(1 - alpha / 2)
    conf_int = critical_value / np.sqrt(n)
    
    conf_int_lower = -conf_int * np.ones_like(acf_values)
    conf_int_upper = conf_int * np.ones_like(acf_values)
    
    # The ACF at lag 0 is always 1, and its confidence interval is not typically plotted around 1.
    conf_int_lower[0] = np.nan
    conf_int_upper[0] = np.nan

    return acf_values, conf_int_lower, conf_int_upper

import plotly.graph_objects as go

def plot_acf(acf_values, conf_int_lower, conf_int_upper, title='Autocorrelation Function'):
    """Plots the Autocorrelation Function (ACF) with confidence intervals.

    Args:
        acf_values (np.ndarray): The ACF values.
        conf_int_lower (np.ndarray): The lower bound of the confidence interval.
        conf_int_upper (np.ndarray): The upper bound of the confidence interval.
        title (str): The title of the plot.
    """
    lags = np.arange(len(acf_values))

    fig = go.Figure()

    # Add ACF values as bars
    fig.add_trace(go.Bar(x=lags, y=acf_values, name='ACF'))

    # Add confidence interval bounds as lines
    fig.add_trace(go.Scatter(x=lags, y=conf_int_upper, mode='lines', line=dict(color='red', dash='dash'), name='Upper Confidence'))
    fig.add_trace(go.Scatter(x=lags, y=conf_int_lower, mode='lines', line=dict(color='red', dash='dash'), name='Lower Confidence'))

    fig.update_layout(title=title, xaxis_title='Lag', yaxis_title='Autocorrelation')
    # fig.show() # This line should be replaced by st.plotly_chart in Streamlit
    return fig # Modified to return figure for Streamlit

def run_page1():
    st.header("Interactive Moving Average (MA) Time Series Analysis")

    st.markdown("""
    This application allows you to explore Moving Average (MA) time series models and their Autocorrelation Function (ACF) interactively.
    """)

    # --- Sidebar for user inputs ---
    with st.sidebar:
        st.header("MA Model Parameters")

        ma_order = st.radio(
            "Select MA Order (q)",
            (1, 2),
            index=0,
            help="Choose the order of the Moving Average model. MA(1) uses one past error term, MA(2) uses two."
        )

        theta_coeffs_list = []
        if ma_order == 1:
            theta1 = st.number_input(
                r"$	heta_1$ (MA(1) Coefficient)",
                min_value=-2.0, max_value=2.0, value=0.7, step=0.1, format="%.2f",
                help="The coefficient determining the influence of the first past error term. Values typically between -1 and 1 for stationarity."
            )
            theta_coeffs_list.append(theta1)
        elif ma_order == 2:
            theta1 = st.number_input(
                r"$	heta_1$ (MA(2) Coefficient)",
                min_value=-2.0, max_value=2.0, value=0.7, step=0.1, format="%.2f",
                help="The coefficient determining the influence of the first past error term. Values typically between -1 and 1 for stationarity."
            )
            theta2 = st.number_input(
                r"$	heta_2$ (MA(2) Coefficient)",
                min_value=-2.0, max_value=2.0, value=0.3, step=0.1, format="%.2f",
                help="The coefficient determining the influence of the second past error term. Values typically between -1 and 1 for stationarity."
            )
            theta_coeffs_list.extend([theta1, theta2])
        
        epsilon_std_dev = st.slider(
            r"White Noise $\epsilon$ Standard Deviation",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help="Controls the volatility of the random error terms. A higher value means more erratic data."
        )

        num_samples = st.slider(
            "Number of Samples (Months)",
            min_value=24, max_value=240, value=100, step=1,
            help="Total number of data points (months) to generate for the time series. Minimum 24 months for 2 years."
        )

        st.header("ACF Plot Settings")
        nlags = st.slider(
            "Number of Lags for ACF Plot",
            min_value=5, max_value=50, value=20, step=1,
            help="Maximum number of lags to display in the Autocorrelation Function plot."
        )

        alpha = st.slider(
            r"Significance Level for ACF ($lpha$)",
            min_value=0.01, max_value=0.10, value=0.05, step=0.01,
            help="Determines the width of the confidence intervals for the ACF. Common values are 0.05 (95% CI) or 0.10 (90% CI)."
        )

    # --- Main content area ---
    st.header("1. Generating MA Data")
    st.markdown(r"""
    This section defines a function to generate synthetic Moving Average (MA) time series data. An MA($q$) model expresses the current value in a time series as a linear combination of the $q$ most recent white noise error terms. The general form of an MA($q$) model is:

    $$x_t = \epsilon_t + 	heta_1 \epsilon_{t-1} + 	heta_2 \epsilon_{t-2} + ... + 	heta_q \epsilon_{t-q}$$

    where:
    *   $x_t$ is the value of the time series at time $t$.
    *   $\epsilon_t$ is a white noise error term at time $t$ (typically assumed to be normally distributed with mean 0 and constant variance).
    *   $	heta_i$ are the MA coefficients that determine the influence of past error terms on the current value.
    """)

    if ma_order == 1:
        st.markdown(r"For example, an MA(1) model is defined as: $x_t = \epsilon_t + 	heta_1 \epsilon_{t-1}$")
        ma_model_eq = r"x_t = \epsilon_t + " + f"{theta_coeffs_list[0]:.2f}" + r"\epsilon_{t-1}"
    elif ma_order == 2:
        st.markdown(r"And an MA(2) model is defined as: $x_t = \epsilon_t + 	heta_1 \epsilon_{t-1} + 	heta_2 \epsilon_{t-2}$")
        ma_model_eq = r"x_t = \epsilon_t + " + f"{theta_coeffs_list[0]:.2f}" + r"\epsilon_{t-1} + " + f"{theta_coeffs_list[1]:.2f}" + r"\epsilon_{t-2}"

    st.markdown(f"**Current MA({ma_order}) Model:**")
    st.latex(ma_model_eq)


    try:
        ma_data = generate_ma_data(ma_order, theta_coeffs_list, epsilon_std_dev, num_samples)
        st.subheader("Generated Time Series")
        fig_ts = go.Figure(data=go.Scatter(x=ma_data.index, y=ma_data.values, mode='lines', name='MA Data'))
        fig_ts.update_layout(
            title=f'Generated MA({ma_order}) Time Series',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode="x unified"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        st.header("2. Calculating the Autocorrelation Function (ACF)")
        st.markdown(r"""
    The Autocorrelation Function (ACF) measures the correlation between a time series and its lagged values. For a time series $x_t$, the ACF at lag $k$ is defined as:

    $$ho_k = rac{Cov(x_t, x_{t-k})}{Var(x_t)}$$

    where:
    *   $Cov(x_t, x_{t-k})$ is the covariance between the time series and its values lagged by $k$.
    *   $Var(x_t)$ is the variance of the time series.

    For an MA($q$) process, the ACF has a characteristic 'cut-off' property. The ACF will be significantly different from zero for lags less than or equal to $q$, and approximately zero for lags greater than $q$.
    """)
        
        acf_values, conf_int_lower, conf_int_upper = calculate_acf(ma_data, nlags=nlags, alpha=alpha)

        st.subheader("Autocorrelation Function Plot")
        acf_plot_fig = plot_acf(acf_values, conf_int_lower, conf_int_upper, title=f'ACF for MA({ma_order}) Process')
        st.plotly_chart(acf_plot_fig, use_container_width=True)

    except ValueError as e:
        st.error(f"Configuration Error: {e}. Please adjust the MA coefficients accordingly.")

    # --- References ---
    st.markdown("---")
    st.header("References")
    st.markdown("""
    *   **Jupyter Notebook Content:** The core functions and conceptual explanations are derived from the provided Jupyter Notebook.
    *   **Libraries:**
        *   Pandas: Data manipulation
        *   NumPy: Numerical operations
        *   SciPy: Statistical functions (for `norm.ppf`)
        *   Plotly: Interactive visualizations
        *   Statsmodels: Time series analysis (`acf`)
        *   Streamlit: Web application framework
    *   **Theoretical Basis:** Concepts adapted from "Quantitative Methods" time-series analysis literature (e.g., CFA Program curriculum).
    """)
