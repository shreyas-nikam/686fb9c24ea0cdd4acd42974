
# Technical Specification for Jupyter Notebook: Moving Average (MA) Model Identifier & Correlogram Visualizer

## 1. Notebook Overview

This Jupyter Notebook provides an interactive environment to explore Moving Average (MA) time series models. It focuses on the defining characteristic of MA models: the behavior of their autocorrelation function (ACF), which is crucial for model identification in time series analysis.

### Learning Goals

Upon completing this lab, users will be able to:
- Understand the fundamental structure and components of Moving Average (MA) models.
- Grasp how the autocorrelation function (ACF) of a time series behaves for MA(q) processes, particularly recognizing the 'cut-off' property.
- Differentiate between Autoregressive (AR) and Moving Average (MA) models based on their distinct ACF patterns.
- Experiment with various MA model parameters ($\theta_1$, $\theta_2$, $\sigma_\epsilon$) and observe their immediate impact on the generated time series and its correlogram.

### Expected Outcomes

Users will interact with a synthetic MA data generator and a correlogram visualizer. They will configure MA model parameters, generate a time series, and then analyze its visual representation and ACF plot. Through this interactive experimentation, users will develop an intuitive understanding of how MA model parameters dictate the memory of the process and its signature ACF pattern, reinforcing a key concept in time series model identification.

## 2. Mathematical and Theoretical Foundations

This section will provide the necessary theoretical background for understanding Moving Average models and their autocorrelation properties, using LaTeX for all mathematical expressions.

### Introduction to Moving Average (MA) Models

A Moving Average (MA) model describes a time series $x_t$ as a linear combination of current and past white noise (error) terms. Unlike Autoregressive (AR) models, MA models do not directly use past observations of the series itself, but rather past forecast errors.

A general Moving Average model of order $q$, denoted as MA($q$), is defined by the equation:

$$x_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}$$

Where:
- $x_t$: The value of the time series at time $t$.
- $\epsilon_t$: The white noise error term at time $t$. This term is assumed to be independently and identically distributed (i.i.d.) with a mean of zero ($E[\epsilon_t] = 0$) and a constant variance of $\sigma^2_\epsilon$. That is, $E[\epsilon_t \epsilon_s] = 0$ for $t \neq s$ and $E[\epsilon_t^2] = \sigma^2_\epsilon$.
- $\theta_1, \theta_2, \dots, \theta_q$: The Moving Average coefficients that define the impact of past error terms on the current value of the series.

For this lab, we will focus on MA(1) and MA(2) models:

**MA(1) Model:**
An MA(1) model incorporates only the most recent past error term:
$$x_t = \epsilon_t + \theta_1 \epsilon_{t-1}$$

**MA(2) Model:**
An MA(2) model includes the two most recent past error terms:
$$x_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2}$$

### Autocorrelation Function (ACF) for MA(q) Models

The Autocorrelation Function (ACF) measures the correlation between a time series and its lagged values. For MA(q) models, the ACF exhibits a distinct property known as the 'cut-off'.

The $k$-th order autocorrelation, $\rho_k$, for a time series is given by:
$$\rho_k = \frac{\text{Cov}(x_t, x_{t-k})}{\text{Var}(x_t)}$$

The sample autocorrelation, $\hat{\rho}_k$, for a given dataset of length $T$ can be estimated as:
$$\hat{\rho}_k = \frac{\sum_{t=k+1}^{T} (x_t - \bar{x})(x_{t-k} - \bar{x})}{\sum_{t=1}^{T} (x_t - \bar{x})^2}$$
where $\bar{x}$ is the sample mean of the series.

**Key Property: The 'Cut-off'**
For an MA($q$) process, the theoretical ACF is non-zero for lags up to $q$ (i.e., $\rho_k \neq 0$ for $k \le q$) and is exactly zero for lags greater than $q$ (i.e., $\rho_k = 0$ for $k > q$). This 'cut-off' property is a defining characteristic of MA models and is essential for their identification.

In practice, with sample data, the estimated ACF values for lags $k > q$ will not be exactly zero but should be statistically insignificant. Statistical significance is typically assessed by comparing the sample autocorrelation $\hat{\rho}_k$ to critical values, often approximated by $\pm \frac{1.96}{\sqrt{T}}$ for a 95% confidence interval, where $T$ is the number of observations. Autocorrelations falling within these bounds are considered not significantly different from zero.

### AR vs. MA Autocorrelation Comparison

It is crucial to distinguish MA models from Autoregressive (AR) models based on their ACF patterns:
- **MA models:** Exhibit a *sharp cut-off* in their ACF after lag $q$. This means autocorrelations are significant for lags $1, \dots, q$ and then become insignificant for all subsequent lags.
- **AR models:** Typically show an ACF that *gradually declines* (either exponentially or sinusoidally) and does not cut off. This gradual decline is a hallmark of AR processes.

This difference in ACF behavior is a primary tool for initial model identification in time series analysis.

## 3. Code Requirements

This section outlines the necessary libraries, expected inputs and outputs, required algorithms, and visualizations for the Jupyter Notebook.

### Expected Libraries

-   `numpy`: For numerical operations, especially for generating random numbers and array manipulations.
-   `pandas`: For handling time series data (e.g., `pd.Series`, `pd.DataFrame`).
-   `statsmodels`: Specifically, `statsmodels.tsa.arima_process.arma_generate_sample` for synthetic MA data generation, and `statsmodels.graphics.tsaplots.plot_acf` for ACF plots.
-   `plotly.graph_objects`: For creating interactive and publication-quality time series and ACF plots.

### Input/Output Expectations

**Input Parameters (via interactive controls):**

1.  **MA Order**: A dropdown menu to select the order of the MA model.
    -   Options: `MA(1)`, `MA(2)`.
    -   Default: `MA(1)`.
    -   Inline help text: "Select the order (q) of the Moving Average model."
2.  **MA Coefficient ($\theta_1$)**: A slider for the $\theta_1$ parameter.
    -   Range: Typically between -1.0 and 1.0 (for invertibility, though not strictly enforced for synthetic data generation in this context, it aligns with common usage).
    -   Step size: 0.01.
    -   Default: 0.5.
    -   Inline help text: "Adjust the coefficient for the $\epsilon_{t-1}$ term. Influences the short-term memory."
3.  **MA Coefficient ($\theta_2$)**: A slider for the $\theta_2$ parameter (only visible/active if MA Order is `MA(2)`).
    -   Range: Typically between -1.0 and 1.0.
    -   Step size: 0.01.
    -   Default: 0.3.
    -   Inline help text: "Adjust the coefficient for the $\epsilon_{t-2}$ term. Relevant for MA(2) models."
4.  **Error Standard Deviation ($\sigma_\epsilon$)**: A slider for the standard deviation of the white noise error term.
    -   Range: e.g., 0.1 to 5.0.
    -   Step size: 0.1.
    -   Default: 1.0.
    -   Inline help text: "Controls the volatility or randomness of the error terms."
5.  **Number of Observations**: A numeric input or slider for the total number of data points.
    -   Minimum: 24 (for 2 years of monthly data).
    -   Default: 100.
    -   Inline help text: "Specify the length of the synthetic time series (at least 2 years of values)."

**Output:**

-   **Generated Time Series Data**: A Pandas Series or DataFrame containing the synthetic MA time series, indexed by time (e.g., date range or simple integer index).
-   **ACF Values**: A Pandas Series or NumPy array of autocorrelation values for different lags.
-   **Confidence Intervals for ACF**: The upper and lower bounds for statistical significance.

### Algorithms or Functions to be Implemented (without code)

1.  **`generate_ma_data(order, theta_coeffs, std_dev, num_observations)`**:
    -   **Purpose**: To simulate a synthetic MA time series based on the specified order and parameters.
    -   **Parameters**:
        -   `order`: Integer (1 or 2) representing the MA order.
        -   `theta_coeffs`: List or array of MA coefficients ($\theta_1$, [$\theta_2$]).
        -   `std_dev`: Float, standard deviation of the white noise error term.
        -   `num_observations`: Integer, the desired length of the time series.
    -   **Internal Logic**:
        -   Create a series of i.i.d. normally distributed random error terms ($\epsilon_t$) with mean 0 and `std_dev`.
        -   Apply the MA equation iteratively or using a dedicated time series generation function (e.g., `statsmodels.tsa.arima_process.arma_generate_sample` where AR coefficients are 0 and MA coefficients are `theta_coeffs`).
        -   Ensure the generated series has at least 2 years of values (e.g., 24 monthly observations).
    -   **Returns**: A Pandas Series representing the generated MA time series.

2.  **`calculate_acf(data, nlags)`**:
    -   **Purpose**: To compute the autocorrelation function of the given time series data.
    -   **Parameters**:
        -   `data`: Pandas Series, the time series for which to calculate ACF.
        -   `nlags`: Integer, the maximum number of lags for ACF calculation.
    -   **Internal Logic**:
        -   Use a statistical function to compute ACF values for specified lags.
        -   Calculate the confidence intervals (e.g., $ \pm 1.96 / \sqrt{T} $).
    -   **Returns**: A tuple containing (ACF values, confidence interval bounds).

### Visualization Requirements

All plots will be generated using `plotly.graph_objects` to enable interactivity.

1.  **Time Series Plot**:
    -   **Type**: Line plot.
    -   **Data**: The generated synthetic MA time series ($x_t$).
    -   **X-axis**: Time index (e.g., 'Time', 'Date').
    -   **Y-axis**: Series values ('$x_t$').
    -   **Title**: "Generated MA(q) Time Series".
    -   **Features**: Interactive zooming and panning.

2.  **Autocorrelation Function (ACF) Plot (Correlogram)**:
    -   **Type**: Bar plot (correlogram).
    -   **Data**: Calculated ACF values for different lags.
    -   **X-axis**: Lag number ('Lag').
    -   **Y-axis**: Autocorrelation coefficient ('Autocorrelation').
    -   **Title**: "Autocorrelation Function (ACF) of Generated MA(q) Series".
    -   **Features**:
        -   Vertical lines representing the ACF values at each lag.
        -   Horizontal dashed lines indicating the 95% confidence intervals for statistical significance ($\pm 1.96 / \sqrt{T}$).
        -   Clear labeling of significant lags and visual identification of the 'cut-off' property.
        -   Interactive tooltips showing exact ACF values and confidence interval bounds on hover.

## 4. Additional Notes or Instructions

-   **Assumptions**: For the purpose of this lab, we assume the underlying error terms ($\epsilon_t$) are white noise, meaning they are independently and identically distributed (i.i.d.) from a normal distribution with zero mean and constant variance.
-   **Interactive Experimentation**: Encourage users to actively adjust the MA order, coefficients ($\theta_1$, $\theta_2$), and error standard deviation. The notebook should be structured so that changing these parameters and re-running subsequent cells immediately updates the time series and ACF plots, facilitating a hands-on learning experience.
-   **Inline Explanations**: Provide clear markdown cells before each code section to explain the purpose of the code and what results to expect. Similarly, after plots, include descriptive text to interpret the visualizations, especially highlighting the 'cut-off' property and comparing MA ACF to AR ACF.
-   **References**: A dedicated "References" section at the end of the notebook, crediting the CFA Institute document and any other relevant sources or libraries used for theoretical explanations or data generation methods. For instance:
    -   [1] CFA Institute. (2024). *REFRESHER READING 2024 CFA® PROGRAM LEVEL 2 Quantitative Methods Time-Series Analysis*. (Specifically, Section "MOVING-AVERAGE TIME-SERIES MODELS" (page 37) and Section "Moving-Average Time-Series Models for Forecasting" (page 39-40)).
```