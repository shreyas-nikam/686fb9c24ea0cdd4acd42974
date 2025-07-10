
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Moving Average Time Series and ARCH Models")
st.divider()
st.markdown("""
In this lab, we delve into the fascinating world of **time series analysis**, focusing on two fundamental models: **Moving Average (MA) processes** and **Autoregressive Conditional Heteroskedasticity (ARCH) models**.

Understanding how time series data behaves is crucial in many fields, from finance and economics to engineering and environmental science. This application provides an interactive platform to:

1.  **Simulate and Visualize Moving Average (MA) Models:** Explore how different parameters (MA coefficients $\theta$ and white noise standard deviation $\sigma$) influence the shape of an MA time series and its Autocorrelation Function (ACF). You'll gain an intuitive understanding of the impact of past error terms on current observations.
2.  **Grasp Autoregressive Conditional Heteroskedasticity (ARCH) Models:** Learn about models that capture changing volatility in financial time series. While the interactive component for ARCH models is a future enhancement, this lab introduces the core concepts and mathematical formulations behind them.

Use the navigation sidebar to explore different aspects of this lab.
""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Moving Average (MA) Models", "ARCH Models Explained", "References"])

if page == "Moving Average (MA) Models":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "ARCH Models Explained":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "References":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
            "Any reproduction of this demonstration "
            "requires prior written consent from QuantUniversity. "
            "This lab was generated using the QuCreate platform. QuCreate relies on AI models for generating code, which may contain inaccuracies or errors")
