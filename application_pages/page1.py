
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf

def generate_ma_data(ma_order, theta, sigma, n_years=2):
    # Calculate the number of data points (assuming daily data)
    n_points = n_years * 365

    # Generate MA data. Note: arma_generate_sample's 'ma' parameter expects
    # coefficients for [noise_t, noise_t-1, ..., noise_t-q], hence 1 at the start.
    ar = np.array([1]) # For MA process, AR coefficients are [1]
    ma = np.concatenate([np.array([1]), theta])  # Ensure 'ma' starts with 1
    noise = np.random.normal(0, sigma, n_points)
    data = arma_generate_sample(ar, ma, nsample=n_points, burnin=100, scale=1, distrvs=lambda n: noise[:n])

    # Create a Pandas Series for time series data
    time_series = pd.Series(data)
    return time_series

def plot_time_series_and_acf(time_series):
    # Time Series Plot
    fig_ts = go.Figure(data=[go.Scatter(y=time_series, mode='lines')])
    fig_ts.update_layout(title='MA Time Series', xaxis_title='Time', yaxis_title='Value')

    # ACF Plot using statsmodels for calculation, then Plotly for visualization
    # sm_plot_acf returns a matplotlib figure, we need to extract data for Plotly
    acf_fig = sm_plot_acf(time_series, lags=30, alpha=0.05) # alpha=0.05 for 95% CI
    # Extract data from the matplotlib figure
    acf_line = acf_fig.axes[0].lines[0]
    lags = acf_line.get_xdata()
    acf_values = acf_line.get_ydata()
    
    # Extract confidence intervals (matplotlib returns these as separate lines)
    # The confidence interval lines are usually the 2nd and 3rd lines in the axes
    lower_ci_line = acf_fig.axes[0].lines[2]
    upper_ci_line = acf_fig.axes[0].lines[1]
    lower_bound = lower_ci_line.get_ydata()[0]
    upper_bound = upper_ci_line.get_ydata()[0]

    # Plotly ACF plot
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Scatter(x=lags, y=acf_values, mode='markers', marker=dict(size=8, color='blue'), name='ACF'))
    fig_acf.add_trace(go.Scatter(x=lags, y=[0]*len(lags), mode='lines', line=dict(color='black', width=1), name='Zero Line', showlegend=False)) # Zero line
    
    fig_acf.add_hline(y=lower_bound, line=dict(color='red', width=1, dash='dash'),
                      annotation_text=f"Lower CI ({lower_bound:.2f})", annotation_position="bottom right",
                      annotation_font_size=10)
    fig_acf.add_hline(y=upper_bound, line=dict(color='red', width=1, dash='dash'),
                      annotation_text=f"Upper CI ({upper_bound:.2f})", annotation_position="top right",
                      annotation_font_size=10)

    fig_acf.update_layout(title='Autocorrelation Function (ACF)',
                          xaxis_title='Lag',
                          yaxis_title='Autocorrelation',
                          showlegend=False)
    
    return fig_ts, fig_acf

def run_page1():
    st.header("Moving Average (MA) Models Simulation")
    st.markdown("""
    A **Moving Average (MA)** model is a common approach for modeling univariate time series. It models the current observation as a linear combination of a white noise error term and past error terms.

    The Moving Average model of order $q$, denoted MA($q$), is defined as:
    $$X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}$$
    where $\epsilon_t$ is a white noise error term with mean $E(\epsilon_t) = 0$ and constant variance $E(\epsilon_t^2) = \sigma^2$, and $E(\epsilon_t \epsilon_s) = 0$ for $t \neq s$.

    Use the controls in the sidebar to adjust the parameters of the MA model and observe their effect on the generated time series and its Autocorrelation Function (ACF).
    """)

    st.sidebar.header("MA Model Parameters")

    ma_order = st.sidebar.radio(
        "Select MA Order (q)",
        options=[1, 2, 3],
        help="The order 'q' determines how many past error terms influence the current value."
    )

    theta_values = []
    if ma_order >= 1:
        theta1 = st.sidebar.slider(
            r"$	heta_1$ (Coefficient for $\epsilon_{t-1}$)",
            min_value=-1.0, max_value=1.0, value=0.7, step=0.01,
            help="The coefficient for the first lagged error term. Controls the direct influence of the most recent shock."
        )
        theta_values.append(theta1)
    if ma_order >= 2:
        theta2 = st.sidebar.slider(
            r"$	heta_2$ (Coefficient for $\epsilon_{t-2}$)",
            min_value=-1.0, max_value=1.0, value=0.3, step=0.01,
            help="The coefficient for the second lagged error term. Influences how shocks from two periods ago affect the current value."
        )
        theta_values.append(theta2)
    if ma_order >= 3:
        theta3 = st.sidebar.slider(
            r"$	heta_3$ (Coefficient for $\epsilon_{t-3}$)",
            min_value=-1.0, max_value=1.0, value=0.1, step=0.01,
            help="The coefficient for the third lagged error term. Extends the influence to shocks from three periods ago."
        )
        theta_values.append(theta3)
    
    sigma = st.sidebar.slider(
        r"$\sigma$ (White Noise Standard Deviation)",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="The standard deviation of the white noise error term. A higher sigma results in more volatile series."
    )

    n_years = st.sidebar.slider(
        "Number of Years (n_years)",
        min_value=2, max_value=10, value=2, step=1,
        help="The length of the simulated time series in years (assuming daily data)."
    )

    st.subheader("Simulated MA Time Series")
    st.markdown(f"**Current Parameters:** MA({ma_order}), "
                f"$\theta$ = {theta_values}, "
                f"$\sigma$ = {sigma}, "
                f"Number of Years = {n_years}")

    # Generate data
    theta_np = np.array(theta_values)
    ma_time_series = generate_ma_data(ma_order, theta_np, sigma, n_years)

    # Plot
    fig_ts, fig_acf = plot_time_series_and_acf(ma_time_series)

    st.plotly_chart(fig_ts, use_container_width=True)
    st.markdown("""
    The **MA Time Series Plot** above shows the simulated data points over time. Notice how the series fluctuates around its mean (which is zero for a pure MA process with zero-mean errors). The degree of fluctuation is influenced by the $\sigma$ (standard deviation of white noise) and the $\theta$ coefficients.
    """)

    st.subheader("Autocorrelation Function (ACF) Plot")
    st.markdown("""
    The **Autocorrelation Function (ACF) Plot** (or Correlogram) displays the correlation of the time series with its own past values (lags).
    The $k$-th order autocorrelation $\rho_k$ is defined as:
    $$\rho_k = \frac{\text{Cov}(X_t, X_{t-k})}{\sigma^2}$$
    For an MA($q$) process, a key theoretical characteristic is that its ACF has significant spikes only up to lag $q$, and then **drops to zero for lags greater than $q$**.

    The red dashed lines represent the 95% confidence intervals. If an autocorrelation spike falls outside these lines, it is considered statistically significant.
    A common rule of thumb for 95% confidence intervals uses $\pm \frac{1.96}{\sqrt{N}}$, where $N$ is the number of observations.
    """)
    st.plotly_chart(fig_acf, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    *   Observe how the ACF plot for an MA($q$) model tends to show significant autocorrelations only for the first $q$ lags, and then the correlations become statistically insignificant (fall within the red dashed lines).
    *   Changing the $\theta$ coefficients will alter the magnitude and sign of the significant autocorrelations.
    *   Increasing $\sigma$ will make the time series more volatile, but it generally doesn't change the *pattern* of the ACF, as the ACF measures relative correlations.
    """)
