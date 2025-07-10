# Streamlit Application Requirements Specification

This document outlines the requirements for developing a Streamlit application based on the provided Jupyter Notebook content and user requirements. It will serve as a blueprint, detailing interactive components and referencing relevant code.

## 1. Application Overview

**Purpose and Objectives:**
The primary purpose of this Streamlit application is to provide an interactive platform for users to understand and experiment with Moving Average (MA) time series models. The application will allow users to dynamically adjust key parameters of MA models and visualize their impact on the generated time series and its Autocorrelation Function (ACF). A secondary objective is to introduce the concept of Autoregressive Conditional Heteroskedasticity (ARCH) models, although interactive functionality for ARCH models is not fully supported by the provided notebook code and will be noted as a future enhancement.

**Learning Outcomes Supported:**
-   Understand the fundamental mechanics of Moving Average (MA) time series models.
-   Visually analyze the characteristics of MA(q) processes, particularly through time series plots and Autocorrelation Function (ACF) plots.
-   Experiment with different MA coefficients ($\theta$) and white noise standard deviation ($\sigma$) to observe their effects on the time series and its ACF.
-   Grasp the concept of Autoregressive Conditional Heteroskedasticity (ARCH) models and their significance in time series analysis (conceptual, non-interactive initially).

## 2. User Interface Requirements

**Layout and Navigation Structure:**
The application will feature a single-page layout, structured with a clear title, a sidebar for user inputs, and a main content area for visualizations and explanations.

**Input Widgets and Controls:**
The sidebar will contain interactive widgets for users to control the MA model parameters:

*   **MA Order Selection:**
    *   A `st.radio` or `st.selectbox` widget to select the order of the MA model ($q$).
    *   Options: MA(1), MA(2), MA(3).
*   **MA Coefficients ($\theta$):**
    *   One or more `st.slider` widgets for the MA coefficients, dynamically displayed based on the selected MA order.
    *   For MA(1): A slider for $\theta_1$.
        *   Range: e.g., -1.0 to 1.0 (typical range for stationarity in MA models, although MA models are always stationary for any $\theta$ values, this range provides good visual experimentation).
        *   Default: $0.7$
    *   For MA(2): Sliders for $\theta_1$ and $\theta_2$.
        *   Range: e.g., -1.0 to 1.0 for both.
        *   Defaults: $\theta_1 = 0.7$, $\theta_2 = 0.3$
    *   For MA(3): Sliders for $\theta_1$, $\theta_2$, and $\theta_3$.
        *   Range: e.g., -1.0 to 1.0 for all.
        *   Defaults: $\theta_1 = 0.7$, $\theta_2 = 0.3$, $\theta_3 = 0.1$
*   **White Noise Standard Deviation ($\sigma$):**
    *   A `st.slider` widget for the standard deviation of the white noise error term.
    *   Range: e.g., 0.1 to 5.0.
    *   Default: $1.0$
*   **Number of Years (`n_years`):**
    *   A `st.slider` or `st.number_input` for the length of the time series.
    *   Range: e.g., 2 to 10 years.
    *   Default: $2$ years (as specified in the notebook, at least 2 years).

**Visualization Components (Charts, Graphs, Tables):**
The main content area will display two interactive Plotly figures:

*   **MA Time Series Plot:**
    *   Displays the generated synthetic MA time series data over time.
    *   X-axis: Time (or data point index).
    *   Y-axis: Value of the time series.
    *   Title: "MA Time Series".
*   **Autocorrelation Function (ACF) Plot (Correlogram):**
    *   Displays the autocorrelation values for different lags of the generated time series.
    *   Includes 95% confidence intervals (red dashed lines) to indicate statistical significance.
    *   X-axis: Lag.
    *   Y-axis: Autocorrelation.
    *   Title: "Autocorrelation Function (ACF)".

**Interactive Elements and Feedback Mechanisms:**
*   All plots should dynamically update as soon as a user changes any parameter in the sidebar.
*   The current values of selected parameters will be displayed alongside the plots for clarity.
*   Informative text will explain the theoretical characteristics of MA models and how the generated plots illustrate these characteristics.
*   A dedicated section for "ARCH Models" will explain the concept, even if interactive elements are not initially present.

## 3. Additional Requirements

*   **Real-time Updates and Responsiveness:** The application must update visualizations in real-time as user inputs (sliders, radio buttons) are adjusted, ensuring a smooth and interactive user experience.
*   **Annotation and Tooltip Specifications:**
    *   Each input widget (sliders for $\theta$, $\sigma$, `n_years`, and the MA Order selector) will have inline help text or tooltips (using `help` parameter in Streamlit widgets) to describe its purpose and impact on the model.
    *   The plots will include clear titles and axis labels.
    *   Explanations adjacent to the plots will interpret the visual outputs, linking them back to the theoretical concepts of MA models and ACF behavior.
*   **References Section:** The application will conclude with a "References" section, crediting the `statsmodels` library and `Plotly`, along with any other external datasets or libraries used in the development (e.g., NumPy, Pandas, SciPy).

## 4. Notebook Content and Code Requirements

This section details the specific Python code and mathematical concepts from the Jupyter Notebook that will be integrated into the Streamlit application.

**4.1. Library Installations (Conceptual)**
The initial `!pip install` command is for environment setup and will not be directly executed within the Streamlit application but implies necessary libraries should be pre-installed in the deployment environment.

**4.2. Core Library Imports:**
The Streamlit application will import the following libraries:

```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf
import streamlit as st # Added for Streamlit app
```

**4.3. MA Data Generation and Plotting Functions:**
The two core functions from the notebook will be directly used.

*   **`generate_ma_data(ma_order, theta, sigma, n_years=2)`:**
    *   **Purpose:** Generates a synthetic Moving Average (MA) time series.
    *   **Integration in Streamlit:** This function will be called whenever the user changes the MA order, $\theta$ coefficients, $\sigma$, or `n_years` inputs.
    *   **Parameters from UI:**
        *   `ma_order`: Integer (1, 2, or 3) from the MA Order selection.
        *   `theta`: A NumPy array of MA coefficient(s) provided by the user via sliders. The length of this array will match `ma_order`.
            *   For MA(1): `theta = np.array([theta1_input])`
            *   For MA(2): `theta = np.array([theta1_input, theta2_input])`
            *   For MA(3): `theta = np.array([theta1_input, theta2_input, theta3_input])`
        *   `sigma`: Float from the `White Noise Standard Deviation` slider.
        *   `n_years`: Integer from the `Number of Years` slider.
    *   **Internal Logic (from Notebook):**
        ```python
        # Extracted from Jupyter Notebook Code Cell
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
        ```
    *   **Mathematical Model Explanation (from Notebook):**
        The Moving Average model of order $q$, denoted MA($q$), is defined as:
        $$X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}$$
        where $\epsilon_t$ is a white noise error term with mean $E(\epsilon_t) = 0$ and constant variance $E(\epsilon_t^2) = \sigma^2$, and $E(\epsilon_t \epsilon_s) = 0$ for $t \neq s$.

*   **`plot_time_series_and_acf(time_series)`:**
    *   **Purpose:** Generates interactive Plotly figures for the time series and its ACF.
    *   **Integration in Streamlit:** This function will be called after `generate_ma_data` produces the time series. The returned Plotly figures will be displayed using `st.plotly_chart()`.
    *   **Parameters from UI:** Takes the `time_series` Pandas Series generated by `generate_ma_data`.
    *   **Internal Logic (from Notebook):**
        ```python
        # Extracted from Jupyter Notebook Code Cell
        def plot_time_series_and_acf(time_series):
            # Time Series Plot
            fig_ts = go.Figure(data=[go.Scatter(y=time_series, mode='lines')])
            fig_ts.update_layout(title='MA Time Series', xaxis_title='Time', yaxis_title='Value')

            # ACF Plot using statsmodels for calculation, then Plotly for visualization
            acf_fig = sm_plot_acf(time_series, lags=30, alpha=0.05) # alpha=0.05 for 95% CI
            acf_trace = acf_fig.axes[0].get_lines()[0]  # Get the line data for ACF values
            lags = acf_trace.get_xdata()
            acf_values = acf_trace.get_ydata()

            # Confidence Interval calculation for 95% CI
            # For a given lag k, the estimated autocorrelation rho_k is significant if |rho_k| > 1.96 / sqrt(N)
            # where N is the number of observations.
            N = len(time_series)
            conf_level = 0.95
            critical_value = norm.ppf(1 - (1 - conf_level) / 2) # Z-score for 95% CI (approx 1.96)
            std_error = 1 / np.sqrt(N)
            lower_bound = -critical_value * std_error
            upper_bound = critical_value * std_error

            # Plotly ACF plot
            fig_acf = go.Figure(data=[go.Scatter(x=lags, y=acf_values, mode='markers', marker=dict(size=8))])
            fig_acf.add_trace(go.Scatter(x=lags, y=[0]*len(lags), mode='lines', line=dict(color='black', width=1), showlegend=False)) # Zero line
            fig_acf.add_hline(y=lower_bound, line=dict(color='red', width=1, dash='dash'), annotation_text=f"Lower CI (-{critical_value:.2f}/sqrt(N))", annotation_position="bottom right")
            fig_acf.add_hline(y=upper_bound, line=dict(color='red', width=1, dash='dash'), annotation_text=f"Upper CI (+{critical_value:.2f}/sqrt(N))", annotation_position="top right")

            fig_acf.update_layout(title='Autocorrelation Function (ACF)',
                                  xaxis_title='Lag',
                                  yaxis_title='Autocorrelation',
                                  showlegend=False)

            return fig_ts, fig_acf
        ```
    *   **ACF Conceptual Explanation (from Notebook):**
        The $k$-th order autocorrelation $\rho_k$ is defined as:
        $$\rho_k = \frac{\text{Cov}(X_t, X_{t-k})}{\sigma^2}$$
        For the error term, $\epsilon_t$:
        $$\rho_{\epsilon,k} = \frac{E(\epsilon_t \epsilon_{t-k})}{\sigma^2}$$
        For an MA($q$) process, the theoretical ACF has significant spikes only up to lag $q$, and then drops to zero for lags greater than $q$. The confidence intervals help to determine if a spike is statistically significant. A common rule of thumb for 95% confidence intervals uses $\pm \frac{1.96}{\sqrt{N}}$, where $N$ is the number of observations.

**4.4. ARCH Models (Conceptual Only from Notebook):**
The provided Jupyter Notebook content introduces ARCH models conceptually but does not contain executable Python code for generating or interactively experimenting with them.

*   **Concept:** Autoregressive Conditional Heteroskedasticity (ARCH) models are used when the variance of the error term in a time series model is not constant but depends on the squared values of previous errors. This phenomenon is called heteroskedasticity.
*   **ARCH(1) Model (from Notebook):**
    The ARCH(1) model describes the conditional variance of the error term $\epsilon_t$ as:
    $$\epsilon_t \sim N(0, \alpha_0 + \alpha_1 \epsilon_{t-1}^2)$$
    where $\alpha_0 > 0$ and $\alpha_1 \ge 0$. If $\alpha_1 = 0$, the variance is constant ($\alpha_0$), indicating no ARCH effects. If $\alpha_1 > 0$, a large error in one period increases the variance of the error in the next period.
*   **Testing for ARCH(1) (from Notebook):**
    ARCH(1) can be tested by regressing the squared residuals ($\hat{\epsilon}_t^2$) from a primary time-series model on a constant and one lag of the squared residuals:
    $$\hat{\epsilon}_t^2 = \alpha_0 + \alpha_1 \hat{\epsilon}_{t-1}^2 + u_t$$
    If the estimated coefficient $\hat{\alpha}_1$ is statistically significant (different from zero), then ARCH(1) effects are present.
*   **Forecasted Variance with ARCH(1) (from Notebook):**
    If ARCH(1) errors exist, the variance of errors in period $t+1$ can be predicted in period $t$ using the formula:
    $$\hat{\sigma}_{t+1}^2 = \hat{\alpha}_0 + \hat{\alpha}_1 \hat{\epsilon}_t^2$$

**4.5. Streamlit Application Flow:**
1.  Set up Streamlit page configuration (`st.set_page_config`).
2.  Create a sidebar for user inputs:
    *   MA Order radio buttons.
    *   Conditional display of $\theta$ sliders based on MA Order.
    *   $\sigma$ slider.
    *   `n_years` slider.
3.  Based on user inputs, call `generate_ma_data` to get `time_series`.
4.  Call `plot_time_series_and_acf` with `time_series` to get `fig_ts` and `fig_acf`.
5.  Display `fig_ts` and `fig_acf` using `st.plotly_chart`.
6.  Add descriptive markdown text explaining the MA model, ACF interpretation, and the significance of parameters.
7.  Include a section for ARCH models, explaining the concept and equations, noting the current lack of interactive code for them.
8.  Add a "References" section.

This specification provides a detailed plan for building the interactive Streamlit application based on the provided content.