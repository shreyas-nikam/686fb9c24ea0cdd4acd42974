
import streamlit as st

def run_page3():
    st.header("References")
    st.markdown("""
    This application utilizes the following libraries:

    *   **Streamlit:** For creating the interactive web application.
    *   **Pandas:** For data manipulation and analysis.
    *   **NumPy:** For numerical computations.
    *   **Plotly:** For creating interactive visualizations.
    *   **SciPy:** For statistical functions (e.g., calculating confidence intervals).
    *   **statsmodels:** For time series analysis functions (e.g., generating MA data and calculating ACF).

    Specifically, the following functions from `statsmodels` were used:

    *   `arma_generate_sample` from `statsmodels.tsa.arima_process`: For generating the MA time series data.
    *   `plot_acf` from `statsmodels.graphics.tsaplots`: For calculating the Autocorrelation Function (ACF).
    """)
