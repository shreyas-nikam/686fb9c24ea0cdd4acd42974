
# Streamlit Application Requirements Specification: Moving Average (MA) Time Series Analysis

This document outlines the requirements for a Streamlit application designed to interactively explore Moving Average (MA) time series models and their Autocorrelation Function (ACF). It is based on the provided Jupyter Notebook content and user requirements.

## 1. Application Overview

**Purpose and Objectives:**
The primary purpose of this Streamlit application is to provide an interactive learning environment for understanding Moving Average (MA) time series models and the behavior of their Autocorrelation Function (ACF). Users will be able to generate synthetic MA data by adjusting model parameters and then visualize the resulting time series and its ACF plot. This interactivity aims to deepen the learner's understanding of how different MA model parameters influence the time series' characteristics and its ACF.

**Specific Objectives:**
*   Enable users to generate synthetic MA(1) and MA(2) time series data.
*   Allow real-time adjustment of MA model parameters (order, coefficients, error variance) and data generation settings (number of samples).
*   Calculate and display the Autocorrelation Function (ACF) of the generated MA series.
*   Visualize the generated time series and its ACF plot, including confidence intervals, using Plotly for interactive graphing.
*   Provide contextual information and mathematical explanations for MA models and ACF.

**Scope Limitation:**
Based on the provided Jupyter Notebook content, this application will focus exclusively on Moving Average (MA) time series models and their Autocorrelation Function (ACF). While the user requirements mention ARCH models, the provided notebook does not contain code for ARCH model implementation, thus it is outside the scope of this particular application build.

## 2. User Interface Requirements

**Layout and Navigation Structure:**
The application will adopt a clean, two-column layout:
*   **Sidebar (Left Column):** Will host all interactive input widgets and controls for model parameters.
*   **Main Content Area (Right Column):** Will display the generated time series plot, the ACF plot, and descriptive text including mathematical formulas.

**Input Widgets and Controls:**
All input controls will be placed in the sidebar to allow users to easily modify parameters and observe changes in real-time.
*   **MA Model Order Selection:**
    *   **Widget Type:** Radio buttons (`st.radio`).
    *   **Label:** "Select MA Order (q)"
    *   **Options:** MA(1), MA(2).
    *   **Default:** MA(1).
    *   **Help Text/Tooltip:** "Choose the order of the Moving Average model. MA(1) uses one past error term, MA(2) uses two."
*   **MA Coefficients (`\theta`):**
    *   **Widget Type:** Numeric input fields (`st.number_input`).
    *   **Labels:** "$\theta_1$ (MA(1) Coefficient)", "$\theta_2$ (MA(2) Coefficient)".
    *   **Visibility:** $\theta_2$ input will only appear if MA(2) order is selected.
    *   **Range/Step:** Flexible range, e.g., -2.0 to 2.0, step 0.1.
    *   **Default:** `[0.7]` for MA(1), `[0.7, 0.3]` for MA(2).
    *   **Help Text/Tooltip:** "The coefficient(s) determining the influence of past error terms. Values typically between -1 and 1 for stationarity."
*   **White Noise Standard Deviation (`\epsilon`):**
    *   **Widget Type:** Slider (`st.slider`).
    *   **Label:** "White Noise $\epsilon$ Standard Deviation"
    *   **Range:** 0.1 to 5.0.
    *   **Default:** 1.0.
    *   **Help Text/Tooltip:** "Controls the volatility of the random error terms. A higher value means more erratic data."
*   **Number of Samples:**
    *   **Widget Type:** Slider (`st.slider`).
    *   **Label:** "Number of Samples (Months)"
    *   **Range:** 24 to 240 (for at least 2 years up to 20 years).
    *   **Default:** 100.
    *   **Help Text/Tooltip:** "Total number of data points (months) to generate for the time series. Minimum 24 months for 2 years."
*   **ACF Lags:**
    *   **Widget Type:** Slider (`st.slider`).
    *   **Label:** "Number of Lags for ACF Plot"
    *   **Range:** 5 to 50.
    *   **Default:** 20.
    *   **Help Text/Tooltip:** "Maximum number of lags to display in the Autocorrelation Function plot."
*   **ACF Significance Level (`\alpha`):**
    *   **Widget Type:** Slider (`st.slider`).
    *   **Label:** "Significance Level for ACF ($\alpha$)"
    *   **Range:** 0.01 to 0.10, step 0.01.
    *   **Default:** 0.05.
    *   **Help Text/Tooltip:** "Determines the width of the confidence intervals for the ACF. Common values are 0.05 (95% CI) or 0.10 (90% CI)."

**Visualization Components (charts, graphs, tables):**
*   **Generated Time Series Plot:**
    *   **Type:** Line chart (Plotly).
    *   **Location:** Main content area.
    *   **Title:** "Generated MA($q$) Time Series" (dynamically updated).
    *   **Axes:** X-axis as time (DatetimeIndex), Y-axis as `x_t`.
*   **Autocorrelation Function (ACF) Plot:**
    *   **Type:** Bar chart with confidence interval lines (Plotly).
    *   **Location:** Main content area, below the time series plot.
    *   **Title:** "Autocorrelation Function (ACF) for MA($q$) Process" (dynamically updated).
    *   **Axes:** X-axis as 'Lag', Y-axis as 'Autocorrelation'.
    *   **Confidence Intervals:** Displayed as dashed red lines.
*   **Dynamic Information Display:**
    *   Display selected model parameters (`q`, $\theta_i$, $\epsilon$ standard deviation, `num_samples`).
    *   Display mathematical definition of the selected MA model.
    *   Display mathematical definition of ACF.
    *   Potentially display a small table of ACF values and confidence intervals for specific lags.

**Interactive Elements and Feedback Mechanisms:**
*   **Real-time Updates:** All plots and displayed information will update automatically as soon as any input parameter is changed by the user.
*   **Input Validation:** The application will provide feedback if `theta_coeffs` for MA(1) or MA(2) do not match the expected number of coefficients for the selected order.
*   **Tooltips/Help Text:** As specified above, inline help text or tooltips will be provided for all interactive controls to guide learners.

## 3. Additional Requirements

*   **Real-time Updates and Responsiveness:** The application will leverage Streamlit's reactive nature to ensure that any change in input parameters immediately triggers the recalculation and re-rendering of the time series and ACF plots, providing instant feedback to the user.
*   **Annotation and Tooltip Specifications:** All input widgets will include clear labels and descriptive tooltips or inline help text (using `st.help` or `st.tooltip` where appropriate) to explain their purpose and expected value ranges, aiding user comprehension.
*   **References Section:** A "References" section will be included at the bottom of the application, crediting the libraries used (e.g., Pandas, NumPy, SciPy, Plotly, Statsmodels) and potentially the CFA Institute material as the conceptual source.

## 4. Notebook Content and Code Requirements

This section details the extracted code from the Jupyter Notebook and how it will be integrated into the Streamlit application.

### Prerequisites (for environment setup):
The following Python libraries are required. They should be installed in the user's environment before running the Streamlit application.
```bash
pip install pandas numpy scipy plotly statsmodels streamlit
```

### Core Functions and Integration:

The application will utilize the three main functions provided in the Jupyter Notebook: `generate_ma_data`, `calculate_acf`, and `plot_acf`.

#### 4.1. `generate_ma_data` Function
This function generates a synthetic Moving Average time series.

**Mathematical Context:**
An MA($q$) model expresses the current value in a time series as a linear combination of the $q$ most recent white noise error terms. The general form of an MA($q$) model is:

$$x_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}$$

where:
*   $x_t$ is the value of the time series at time $t$.
*   $\epsilon_t$ is a white noise error term at time $t$ (typically assumed to be normally distributed with mean 0 and constant variance).
*   $\theta_i$ are the MA coefficients that determine the influence of past error terms on the current value.

For example, an MA(1) model is defined as: $x_t = \epsilon_t + \theta_1 \epsilon_{t-1}$
And an MA(2) model is defined as: $x_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2}$

**Code from Jupyter Notebook:**
```python
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
        raise ValueError("Order must be 1 or 2.") # This line is technically redundant due to the initial check.

    # Create pandas Series with DatetimeIndex
    index = pd.date_range(start='2023-01-01', periods=num_samples, freq='M')
    ma_series = pd.Series(ma_data, index=index)
    
    return ma_series
```

**Streamlit Integration:**
*   The `order` parameter will be sourced from an `st.radio` widget.
*   `theta_coeffs` will be a list constructed from one or two `st.number_input` widgets, depending on the selected `order`. Input validation will be implemented to ensure the correct number of coefficients are provided.
*   `epsilon_std_dev` will be controlled by an `st.slider`.
*   `num_samples` will be controlled by another `st.slider`.
*   The output `ma_series` will be used as input for the `calculate_acf` function and for plotting the time series.

#### 4.2. `calculate_acf` Function
This function calculates the Autocorrelation Function (ACF) and its confidence intervals for a given time series.

**Mathematical Context:**
The Autocorrelation Function (ACF) measures the correlation between a time series and its lagged values. For a time series $x_t$, the ACF at lag $k$ is defined as:

$$\rho_k = \frac{Cov(x_t, x_{t-k})}{Var(x_t)}$$

where:
*   $Cov(x_t, x_{t-k})$ is the covariance between the time series and its values lagged by $k$.
*   $Var(x_t)$ is the variance of the time series.

**Code from Jupyter Notebook:**
```python
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
```

**Streamlit Integration:**
*   The `series` input will be the `ma_series` returned from `generate_ma_data`.
*   `nlags` will be controlled by an `st.slider` widget.
*   `alpha` will be controlled by an `st.slider` widget.
*   The three output arrays (`acf_values`, `conf_int_lower`, `conf_int_upper`) will be passed to the `plot_acf` function.

#### 4.3. `plot_acf` Function
This function visualizes the ACF with confidence intervals using Plotly.

**Code from Jupyter Notebook:**
```python
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
```

**Streamlit Integration:**
*   The input arrays will be the outputs from `calculate_acf`.
*   The `title` will be dynamically generated based on the selected MA order.
*   The function will be slightly modified to *return* the Plotly figure object, which then `st.plotly_chart()` will display in the main content area.

### Overall Streamlit Application Flow:

1.  **Imports:** Import `streamlit` (`st`), `pandas`, `numpy`, `plotly.graph_objects`, `statsmodels.tsa.stattools`, `scipy.stats`.
2.  **Page Configuration:** Set page title and layout (`st.set_page_config`).
3.  **Application Title:** Display main title "Interactive Moving Average (MA) Time Series Analysis".
4.  **Sidebar Configuration:**
    *   **MA Model Order:** Radio button for MA(1) or MA(2).
    *   **Conditional Theta Inputs:** Dynamically show `st.number_input` for $\theta_1$ and $\theta_2$ based on the selected MA order. Implement validation logic.
    *   **Epsilon Std Dev:** Slider for `epsilon_std_dev`.
    *   **Number of Samples:** Slider for `num_samples`.
    *   **ACF Lags:** Slider for `nlags`.
    *   **ACF Alpha:** Slider for `alpha`.
5.  **Main Content Area - Model Explanation:**
    *   Display Markdown text for MA model mathematical definitions using `st.markdown` with LaTeX.
6.  **Data Generation and Analysis Logic:**
    *   Call `generate_ma_data` with user-selected parameters. Wrap in a `try-except` block to catch `ValueError` from coefficient mismatch.
    *   If data generation is successful, display the generated time series using `st.line_chart` (for simplicity or `st.plotly_chart` with `go.Scatter`).
    *   Call `calculate_acf` with the generated series and user-selected lags/alpha.
    *   Call `plot_acf` with the ACF results.
    *   Display the ACF plot using `st.plotly_chart`.
7.  **Additional Explanations:** Display Markdown text for ACF mathematical definitions.
8.  **References Section:** Conclude with a "References" section using `st.markdown`.

```python
# Example of Streamlit logic structure (not the full app code, just the flow)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf
from scipy.stats import norm

# --- Function definitions (as provided in the notebook, possibly with minor return changes for Plotly) ---
# def generate_ma_data(...): ...
# def calculate_acf(...): ...
# def plot_acf(...): ... (modified to return fig)

st.set_page_config(layout="wide", page_title="MA Time Series & ACF")

st.title("Interactive Moving Average (MA) Time Series Analysis")

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
            r"$\theta_1$ (MA(1) Coefficient)",
            min_value=-2.0, max_value=2.0, value=0.7, step=0.1, format="%.2f",
            help="The coefficient determining the influence of the first past error term. Values typically between -1 and 1 for stationarity."
        )
        theta_coeffs_list.append(theta1)
    elif ma_order == 2:
        theta1 = st.number_input(
            r"$\theta_1$ (MA(2) Coefficient)",
            min_value=-2.0, max_value=2.0, value=0.7, step=0.1, format="%.2f",
            help="The coefficient determining the influence of the first past error term. Values typically between -1 and 1 for stationarity."
        )
        theta2 = st.number_input(
            r"$\theta_2$ (MA(2) Coefficient)",
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
        r"Significance Level for ACF ($\alpha$)",
        min_value=0.01, max_value=0.10, value=0.05, step=0.01,
        help="Determines the width of the confidence intervals for the ACF. Common values are 0.05 (95% CI) or 0.10 (90% CI)."
    )

# --- Main content area ---
st.header("1. Generating MA Data")
st.markdown(r"""
This section defines a function to generate synthetic Moving Average (MA) time series data. An MA($q$) model expresses the current value in a time series as a linear combination of the $q$ most recent white noise error terms. The general form of an MA($q$) model is:

$$x_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}$$

where:
*   $x_t$ is the value of the time series at time $t$.
*   $\epsilon_t$ is a white noise error term at time $t$ (typically assumed to be normally distributed with mean 0 and constant variance).
*   $\theta_i$ are the MA coefficients that determine the influence of past error terms on the current value.
""")

if ma_order == 1:
    st.markdown(r"For example, an MA(1) model is defined as: $x_t = \epsilon_t + \theta_1 \epsilon_{t-1}$")
    ma_model_eq = r"x_t = \epsilon_t + " + f"{theta_coeffs_list[0]:.2f}" + r"\epsilon_{t-1}"
elif ma_order == 2:
    st.markdown(r"And an MA(2) model is defined as: $x_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2}$")
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

$$\rho_k = \frac{Cov(x_t, x_{t-k})}{Var(x_t)}$$

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

```
