
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we will explore Moving Average (MA) time series models and their Autocorrelation Function (ACF). You can interactively generate MA data by adjusting model parameters and visualize the resulting time series and its ACF plot.
""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["MA Model and ACF"])
if page == "MA Model and ACF":
    from application_pages.page1 import run_page1
    run_page1()
# Your code ends
st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
            "Any reproduction of this demonstration "
            "requires prior written consent from QuantUniversity. "
            "This lab was generated using the QuCreate platform. QuCreate relies on AI models for generating code, which may contain inaccuracies or errors")
