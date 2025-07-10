
# Technical Specification for Jupyter Notebook: Moving Average (MA) Model Identifier & Correlogram Visualizer

This document specifies the design and content of a Jupyter Notebook focused on illustrating the characteristics of Moving Average (MA) time series models, particularly their Autocorrelation Function (ACF) patterns.

---

## 1. Notebook Overview

### 1.1. Learning Goals

The primary learning goals for users interacting with this notebook are:

*   To understand the fundamental structure and definition of Moving Average (MA) models.
*   To learn how the Autocorrelation Function (ACF) of a time series behaves specifically for MA(q) processes, with a focus on identifying the 'cut-off' property.
*   To develop the ability to differentiate between Autoregressive (AR) and Moving Average (MA) models based on the distinct patterns observed in their ACF plots.
*   To gain practical experience by experimenting with various MA model parameters and observing their immediate impact on both the generated time series and its corresponding autocorrelations.

### 1.2. Expected Outcomes

Upon completion of this lab, users are expected to:

*   Be able to generate synthetic MA(1) and MA(2) time series data with user-defined parameters.
*   Successfully visualize the generated time series data and its autocorrelation function (correlogram).
*   Interact effectively with model parameters (MA coefficients and error standard deviation) and interpret the dynamic changes in the time series and ACF plots.
*   Articulate the significance of the 'cut-off' property in the ACF for MA models as a key tool for model identification.

---

## 2. Mathematical and Theoretical Foundations

This section provides the core mathematical definitions and theoretical background necessary for understanding Moving Average models and their autocorrelation properties.

### 2.1. Moving Average (MA) Models

A Moving Average (MA) model of order $q$, denoted as MA(q), expresses the current value of a time series as a linear combination of the current and past white noise (error) terms. Unlike Autoregressive (AR) models which depend on past values of the series itself, MA models depend on past forecast errors.

#### 2.1.1. General Form of MA(q)

A time series $x_t$ follows an MA(q) process if it can be written as:

$$ x_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q} $$

Where:
*   $x_t$ is the value of the time series at time $t$.
*   $\mu$ is the mean of the series (often assumed to be zero for simplicity in theoretical examples, especially when focusing on deviations).
*   $\epsilon_t$ is a white noise error term at time $t$. White noise is characterized by:
    *   Mean: $E[\epsilon_t] = 0$
    *   Constant Variance: $E[\epsilon_t^2] = \sigma_{\epsilon}^2$ (finite and positive)
    *   No Autocorrelation: $E[\epsilon_t \epsilon_s] = 0$ for $t \neq s$
*   $\theta_1, \theta_2, \dots, \theta_q$ are the moving average coefficients.

#### 2.1.2. Specific Forms for MA(1) and MA(2)

As per the input, the focus will be on MA(1) and MA(2) models, typically presented without a mean term ($\mu=0$) when illustrating their pure error-term dependence:

*   **MA(1) Model**:
    $$ x_t = \epsilon_t + \theta_1 \epsilon_{t-1} $$
    This model states that the current value of $x_t$ depends on the current white noise error $\epsilon_t$ and the previous period's white noise error $\epsilon_{t-1}$.

*   **MA(2) Model**:
    $$ x_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} $$
    Here, $x_t$ depends on the current white noise error $\epsilon_t$ and the white noise errors from the two previous periods, $\epsilon_{t-1}$ and $\epsilon_{t-2}$.

### 2.2. Autocorrelation Function (ACF)

The Autocorrelation Function (ACF) measures the linear relationship between a time series's current value and its past values. The $k$-th order autocorrelation, $\rho_k$, quantifies the correlation between $x_t$ and $x_{t-k}$.

#### 2.2.1. Definition of $k$-th Order Autocorrelation

The theoretical $k$-th order autocorrelation, $\rho_k$, for a stationary time series is defined as:

$$ \rho_k = \frac{\text{Cov}(x_t, x_{t-k})}{\text{Var}(x_t)} $$

Where:
*   $\text{Cov}(x_t, x_{t-k})$ is the covariance between the series at time $t$ and at time $t-k$.
*   $\text{Var}(x_t)$ is the variance of the series at time $t$.

#### 2.2.2. The 'Cut-Off' Property of MA(q) Models

A defining characteristic of MA(q) models is their "cut-off" property in the ACF:

*   For an MA(q) process, the autocorrelation coefficients $\rho_k$ will be non-zero for lags $k$ up to the order $q$ of the model.
*   Crucially, for all lags $k$ greater than $q$, the autocorrelation coefficients $\rho_k$ will be exactly zero.

This property means that an MA(q) process only has "memory" of $q$ past error terms.

#### 2.2.3. Derivation for MA(1) ACF (Illustrative Example)

Let's derive the first few autocorrelations for an MA(1) model, $x_t = \epsilon_t + \theta_1 \epsilon_{t-1}$.

First, calculate the variance of $x_t$:
$$ \text{Var}(x_t) = E[(x_t - E[x_t])^2] $$
Since $E[x_t] = E[\epsilon_t + \theta_1 \epsilon_{t-1}] = E[\epsilon_t] + \theta_1 E[\epsilon_{t-1}] = 0 + \theta_1 \cdot 0 = 0$:
$$ \text{Var}(x_t) = E[(\epsilon_t + \theta_1 \epsilon_{t-1})^2] = E[\epsilon_t^2 + 2\theta_1 \epsilon_t \epsilon_{t-1} + \theta_1^2 \epsilon_{t-1}^2] $$
Given that white noise errors are uncorrelated ($E[\epsilon_t \epsilon_{t-1}] = 0$):
$$ \text{Var}(x_t) = E[\epsilon_t^2] + \theta_1^2 E[\epsilon_{t-1}^2] = \sigma_{\epsilon}^2 + \theta_1^2 \sigma_{\epsilon}^2 = \sigma_{\epsilon}^2 (1 + \theta_1^2) $$

Next, calculate the first-order covariance, $\text{Cov}(x_t, x_{t-1})$:
$$ \text{Cov}(x_t, x_{t-1}) = E[(x_t - E[x_t])(x_{t-1} - E[x_{t-1}])] $$
Since $E[x_t] = 0$ and $E[x_{t-1}] = 0$:
$$ \text{Cov}(x_t, x_{t-1}) = E[(\epsilon_t + \theta_1 \epsilon_{t-1}) (\epsilon_{t-1} + \theta_1 \epsilon_{t-2})] $$
$$ = E[\epsilon_t \epsilon_{t-1} + \theta_1 \epsilon_t \epsilon_{t-2} + \theta_1 \epsilon_{t-1}^2 + \theta_1^2 \epsilon_{t-1} \epsilon_{t-2}] $$
Due to the white noise properties ($E[\epsilon_i \epsilon_j] = 0$ for $i \neq j$):
$$ \text{Cov}(x_t, x_{t-1}) = \theta_1 E[\epsilon_{t-1}^2] = \theta_1 \sigma_{\epsilon}^2 $$

Finally, compute the first-order autocorrelation, $\rho_1$:
$$ \rho_1 = \frac{\text{Cov}(x_t, x_{t-1})}{\text{Var}(x_t)} = \frac{\theta_1 \sigma_{\epsilon}^2}{\sigma_{\epsilon}^2 (1 + \theta_1^2)} = \frac{\theta_1}{1 + \theta_1^2} $$

Now, consider the second-order covariance, $\text{Cov}(x_t, x_{t-2})$:
$$ \text{Cov}(x_t, x_{t-2}) = E[(\epsilon_t + \theta_1 \epsilon_{t-1}) (\epsilon_{t-2} + \theta_1 \epsilon_{t-3})] $$
$$ = E[\epsilon_t \epsilon_{t-2} + \theta_1 \epsilon_t \epsilon_{t-3} + \theta_1 \epsilon_{t-1} \epsilon_{t-2} + \theta_1^2 \epsilon_{t-1} \epsilon_{t-3}] $$
All these terms involve uncorrelated white noise errors, so:
$$ \text{Cov}(x_t, x_{t-2}) = 0 $$
Therefore, for an MA(1) model, $\rho_2 = 0$. By extension, $\rho_k = 0$ for all $k > 1$. This derivation clearly demonstrates the "cut-off" property.

### 2.3. AR vs. MA Autocorrelation Patterns

The ACF is a crucial tool for identifying the order of AR and MA models.

*   **Autoregressive (AR) Models**: For an AR(p) model, the ACF typically *tails off* gradually, either decaying exponentially or in a sinusoidal (damped sine wave) pattern. It does not suddenly drop to zero.
*   **Moving Average (MA) Models**: For an MA(q) model, the ACF *cuts off* abruptly to zero after lag $q$. This distinct pattern is the primary characteristic used to identify MA models.

This notebook will allow direct observation of this cut-off property for MA(1) and MA(2) models by varying their parameters.

---

## 3. Code Requirements

This section details the necessary libraries, input/output structures, and the functional components for the interactive Jupyter Notebook. No actual Python code will be written here, only their specifications.

### 3.1. Expected Libraries

The following Python libraries are expected to be used for implementing the features:

*   **`numpy`**: For numerical operations, especially for generating random numbers (white noise error terms) and array manipulations required for time series calculations.
*   **`pandas`**: For efficient handling and manipulation of time series data, including creating and managing a `DatetimeIndex`.
*   **`statsmodels.tsa.arima_process`**: Specifically, `arma_generate_sample` for generating synthetic MA time series data.
*   **`statsmodels.graphics.tsaplots`**: Specifically, `plot_acf` for plotting the autocorrelation function.
*   **`plotly.graph_objects`**: For creating interactive and publication-quality time series and ACF plots.
*   **`ipywidgets`**: For building interactive user controls such as dropdowns and sliders, enabling dynamic parameter adjustment.
*   **`IPython.display`**: For displaying the interactive widgets and updating output dynamically within the notebook.

### 3.2. Input/Output Expectations

#### 3.2.1. User Input Controls

The notebook will feature interactive controls implemented using `ipywidgets`. These controls will allow users to configure the MA model parameters and observe their effects in real-time.

*   **Model Order Selection (Dropdown)**:
    *   **Control Type**: Dropdown menu.
    *   **Label**: "Select MA Order (q)".
    *   **Options**: `1` (for MA(1)) and `2` (for MA(2)).
    *   **Default Value**: `1`.
    *   **Help Text**: "Choose the order of the Moving Average model."

*   **MA Coefficient Sliders ($\theta_1, \theta_2$)**:
    *   **Control Type**: Float sliders.
    *   **$\theta_1$ Slider**:
        *   **Label**: "Theta 1 ($\theta_1$)".
        *   **Range**: -1.0 to 1.0 (e.g., step 0.05).
        *   **Default Value**: 0.5.
        *   **Help Text**: "Coefficient for the first lagged error term ($\epsilon_{t-1}$). Controls the influence of the immediate past error."
    *   **$\theta_2$ Slider**:
        *   **Label**: "Theta 2 ($\theta_2$)".
        *   **Range**: -1.0 to 1.0 (e.g., step 0.05).
        *   **Default Value**: 0.2.
        *   **Help Text**: "Coefficient for the second lagged error term ($\epsilon_{t-2}$). Only active for MA(2) models. Controls the influence of the error from two periods ago."
        *   **Conditional Behavior**: This slider must be disabled or hidden when MA Order is set to 1.

*   **Error Standard Deviation Slider ($\sigma_{\epsilon}$)**:
    *   **Control Type**: Float slider.
    *   **Label**: "Error Std Dev ($\sigma_{\epsilon}$)".
    *   **Range**: 0.1 to 2.0 (e.g., step 0.1).
    *   **Default Value**: 1.0.
    *   **Help Text**: "Standard deviation of the white noise error term ($\epsilon_t$). A higher value increases the volatility of the series."

#### 3.2.2. Output Visualizations

The notebook will dynamically generate and display two interactive plots based on the selected parameters.

*   **Plot 1: Generated MA Time Series**
    *   **Type**: Interactive line plot (using Plotly).
    *   **Purpose**: Displays the synthetic MA time series data generated with the chosen parameters.
    *   **X-axis**: Time (represented by a `DatetimeIndex`, e.g., monthly dates covering at least 2 years).
    *   **Y-axis**: Value of the generated MA series ($x_t$).
    *   **Interactive Features**: Zooming, panning, and hover-over information for data points.

*   **Plot 2: Autocorrelation Function (ACF) Plot (Correlogram)**
    *   **Type**: Interactive bar plot (using Plotly, ideally leveraging `statsmodels` plotting capabilities).
    *   **Purpose**: Visualizes the autocorrelation coefficients of the generated MA time series at various lags.
    *   **X-axis**: Lag (e.g., from 1 up to a reasonable number like 10-20 lags to clearly show the cut-off).
    *   **Y-axis**: Autocorrelation coefficient ($\rho_k$).
    *   **Key Elements**:
        *   Bars representing the autocorrelation value for each lag.
        *   Horizontal dashed lines representing the 95% confidence intervals (critical values) for statistical significance (typically at $\pm 1.96 / \sqrt{N}$, where $N$ is the number of observations). Autocorrelations falling outside these lines are considered statistically significant.
    *   **Interactive Features**: Zooming, panning, and hover-over information for bars.

### 3.3. Algorithms or Functions to be Implemented

The core logic will be encapsulated in Python functions, but their internal implementation code is not part of this specification.

*   **`generate_ma_data(order, theta_coeffs, epsilon_std_dev, num_samples)`**:
    *   **Description**: This function will generate a synthetic Moving Average time series.
    *   **Inputs**:
        *   `order` (int): The order of the MA model (1 or 2).
        *   `theta_coeffs` (list of float): A list of MA coefficients [$\theta_1$, $\theta_2$, ...]. The length of this list will depend on the `order`.
        *   `epsilon_std_dev` (float): The standard deviation ($\sigma_{\epsilon}$) of the white noise error terms.
        *   `num_samples` (int): The desired number of data points for the time series (e.g., 24 for 2 years of monthly data).
    *   **Process**:
        1.  Generate a sequence of `num_samples` white noise error terms $\epsilon_t$ using a normal distribution with mean 0 and standard deviation `epsilon_std_dev`.
        2.  Construct the MA series $x_t$ based on the selected `order` and `theta_coeffs` using the MA model equation. For instance, for MA(1), $x_t = \epsilon_t + \theta_1 \epsilon_{t-1}$. For MA(2), $x_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2}$. Initial values for lagged errors will be set to zero.
        3.  Create a `pandas.Series` with a `DatetimeIndex` (e.g., monthly frequency) for the generated series.
    *   **Returns**: A `pandas.Series` object representing the synthetic MA time series.

*   **`calculate_acf(data, lags)`**:
    *   **Description**: This function will compute the autocorrelation function (ACF) for a given time series.
    *   **Inputs**:
        *   `data` (pandas.Series): The time series for which to calculate the ACF.
        *   `lags` (int): The maximum number of lags for which to compute the ACF.
    *   **Process**: Utilize `statsmodels.graphics.tsaplots.plot_acf` internally to get the ACF values and confidence intervals without plotting directly, or use `statsmodels.tsa.stattools.acf` and calculate confidence intervals manually.
    *   **Returns**: A tuple containing:
        *   An array of ACF values for each lag.
        *   An array of confidence intervals (upper and lower bounds).

*   **`plot_time_series(data)`**:
    *   **Description**: Creates an interactive line plot of the time series data using Plotly.
    *   **Inputs**:
        *   `data` (pandas.Series): The time series data to plot.
    *   **Process**:
        1.  Initialize a Plotly `Figure`.
        2.  Add a line trace for the `data`.
        3.  Set appropriate layout, including x-axis label ("Time" or "Date") and y-axis label ("MA Value").
        4.  Ensure interactivity (zoom, pan, hover) is enabled.
    *   **Returns**: A Plotly `Figure` object.

*   **`plot_acf(acf_values, conf_int, title)`**:
    *   **Description**: Generates an interactive correlogram (ACF plot) using Plotly.
    *   **Inputs**:
        *   `acf_values` (numpy.array): The autocorrelation coefficients.
        *   `conf_int` (numpy.array): The confidence interval bounds (e.g., from `statsmodels.graphics.tsaplots.plot_acf`).
        *   `title` (str): Title for the ACF plot.
    *   **Process**:
        1.  Initialize a Plotly `Figure`.
        2.  Add bar traces for the `acf_values` (lag vs. correlation).
        3.  Add horizontal line traces for the upper and lower confidence bounds. These lines should be styled differently (e.g., dashed, lighter color) and clearly indicate significance thresholds.
        4.  Set appropriate layout, including x-axis label ("Lag") and y-axis label ("Autocorrelation").
        5.  Ensure interactivity (zoom, pan, hover) is enabled.
    *   **Returns**: A Plotly `Figure` object.

*   **`update_plots(ma_order, theta_1, theta_2, epsilon_std_dev)`**:
    *   **Description**: This is the main callback function connected to the `ipywidgets`. It orchestrates data generation and plotting.
    *   **Inputs**: Parameters directly from the interactive widgets.
    *   **Process**:
        1.  Determine the `theta_coeffs` list based on `ma_order`. If `ma_order` is 1, `theta_coeffs` will be `[theta_1]`. If `ma_order` is 2, `theta_coeffs` will be `[theta_1, theta_2]`.
        2.  Call `generate_ma_data` with the current parameters.
        3.  Call `calculate_acf` on the generated data.
        4.  Call `plot_time_series` and `plot_acf` with the results.
        5.  Display both Plotly figures using `IPython.display.display`.
    *   **Returns**: None (outputs plots directly).

---

## 4. Additional Notes or Instructions

### 4.1. Assumptions

*   The underlying white noise error term ($\epsilon_t$) for the MA models is assumed to be independently and identically distributed (i.i.d.) from a normal distribution with a mean of zero and constant variance.
*   The chosen MA coefficients ($\theta_1$, $\theta_2$) will typically be within the invertibility region for realistic time series behavior, although the data generation process itself does not strictly enforce this for the purpose of demonstrating ACF properties.
*   The generated synthetic dataset will consist of monthly values, spanning a minimum of 2 years (i.e., at least 24 data points), ensuring sufficient observations for ACF computation and visualization.

### 4.2. Constraints

*   The notebook's scope is strictly limited to Moving Average (MA) models of order 1 (MA(1)) and order 2 (MA(2)), as specified in the requirements. No other time series models (e.g., AR, ARMA, ARIMA) will be covered or implemented.
*   No deployment-specific instructions or code (e.g., for Streamlit, Dash, etc.) will be included. The notebook is designed for local execution within a Jupyter environment.
*   All mathematical content must strictly adhere to LaTeX formatting rules: `$$...$$` for display equations (centered, on their own line) and `$...$` for inline equations (within text).
*   No executable Python code should be written directly within this specification document, only function signatures and conceptual descriptions.

### 4.3. Customization and User Interaction

*   **Dynamic Updates**: All visualizations (time series plot and ACF plot) must update dynamically and instantaneously when the user adjusts any of the interactive parameters (MA order, $\theta_1$, $\theta_2$, $\sigma_{\epsilon}$).
*   **Inline Help/Tooltips**: Each interactive control (dropdown, sliders) will have clear inline help text or tooltips to explain its purpose and effect on the model and data.
*   **Interpretive Text**: Markdown cells will be strategically placed to guide the user's interpretation of the observed time series and ACF patterns, emphasizing the core concepts (e.g., the MA cut-off property).
*   **References**: A dedicated "References" section will be included at the end of the notebook, crediting the CFA Institute document and any other external resources used.

---

### References

*   [1] User instructions for learning outcomes.
*   [8] Section "MOVING-AVERAGE TIME-SERIES MODELS" (page 37) and Section "Moving-Average Time-Series Models for Forecasting" (page 39-40), "REFRESHER READING 2024 CFA® PROGRAM LEVEL 2 Quantitative Methods Time-Series Analysis", CFA Institute Document.

