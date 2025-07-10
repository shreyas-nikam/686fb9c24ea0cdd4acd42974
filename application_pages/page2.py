
import streamlit as st

def run_page2():
    st.header("Understanding ARCH Models")
    st.markdown("""
    **Autoregressive Conditional Heteroskedasticity (ARCH) models** are used to model time series data where the variance of the error term changes over time.
    In simpler terms, ARCH models are useful when the volatility of a time series is not constant but depends on the past values of the series.
    This is common in financial time series, where periods of high volatility tend to cluster together.

    **Key Concepts:**

    *   **Heteroskedasticity:** The variance of the error term is not constant.
    *   **Conditional Variance:** The variance of the error term depends on past values.
    *   **Volatility Clustering:** Periods of high volatility tend to be followed by periods of high volatility, and vice versa.

    **ARCH(1) Model:**

    The ARCH(1) model is the simplest form of ARCH model. It assumes that the conditional variance of the error term $\epsilon_t$ depends on the squared value of the previous error term $\epsilon_{t-1}$.
    The ARCH(1) model is defined as follows:

    $$\epsilon_t \sim N(0, \alpha_0 + \alpha_1 \epsilon_{t-1}^2)$$

    where:
    *   $\epsilon_t$ is the error term at time $t$.
    *   $\alpha_0 > 0$ is the constant term.
    *   $\alpha_1 \geq 0$ is the coefficient that determines the sensitivity of the conditional variance to the previous error term.
    *   $\epsilon_{t-1}$ is the error term at time $t-1$.

    **Interpretation:**

    *   If $\alpha_1 = 0$, the variance is constant ($\alpha_0$), indicating no ARCH effects.
    *   If $\alpha_1 > 0$, a large error in one period increases the variance of the error in the next period.

    **Testing for ARCH(1) Effects:**

    ARCH(1) effects can be tested by regressing the squared residuals ($\\hat{\epsilon}_t^2$) from a primary time-series model on a constant and one lag of the squared residuals:

    $$\\hat{\epsilon}_t^2 = \\alpha_0 + \\alpha_1 \\hat{\epsilon}_{t-1}^2 + u_t$$

    If the estimated coefficient $\\hat{\\alpha}_1$ is statistically significant (different from zero), then ARCH(1) effects are present.

    **Forecasted Variance with ARCH(1):**

    If ARCH(1) errors exist, the variance of errors in period $t+1$ can be predicted in period $t$ using the formula:

    $$\\hat{\sigma}_{t+1}^2 = \\hat{\\alpha}_0 + \\hat{\\alpha}_1 \\hat{\epsilon}_t^2$$

    **Limitations:**

    The current implementation of this lab does not include interactive visualizations or simulations for ARCH models. This is planned as a future enhancement.
    """)
