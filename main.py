import streamlit as st
import pandas as pd
import numpy as np
from ui import render_step_navigation, render_step1_data_upload, render_step2_model_configuration, render_step3_results

def main():
    st.set_page_config(layout="wide")
    st.title("Time Series Forecasting App")
    
    # Render step navigation and get current step
    current_step = render_step_navigation()
    
    # Render the appropriate step based on navigation
    if current_step == 1:
        render_step1_data_upload()
    elif current_step == 2:
        render_step2_model_configuration()
    elif current_step == 3:
        render_step3_results()

if __name__ == "__main__":
    main()
