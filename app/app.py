import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.dates as mdates
import os,sys


rpath = os.path.abspath('..')
if rpath not in sys.path:
    sys.path.insert(0, rpath)


#from scripts.utils import *
from scripts.utils import *



def main():
    st.title("Solar Radiation EDA Visualization")

    # Define file paths
    file_paths = {
        'Benin': '../data/benin-malanville.csv',
        'Sierra_Leone': '../data/sierraleone-bumbuna.csv',
        'Togo': '../data/togo-dapaong_qc.csv'
    }

    # Load data
    dfs = load_data(file_paths)

    # Clean and preprocess data
    for country, df in dfs.items():
        dfs[country] = convert_timestamp(df)
        dfs[country] = handle_negative_values(df, country)

    cleaned_dfs = clean_data(dfs)

    # Sidebar for country selection
    country = st.sidebar.selectbox("Select a country", list(cleaned_dfs.keys()))



    # Time Series Analysis
    if st.sidebar.checkbox("Show Time Series Analysis"):
        st.subheader("Time Series Analysis")
        columns = st.multiselect("Select columns for Time Series", 
                                 cleaned_dfs[country].columns)
        if columns:
            fig=time_series({country: cleaned_dfs[country]}, columns)
            st.pyplot(fig)

    # Correlation Heatmap
    if st.sidebar.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        fig=plot_correlation_heatmap(cleaned_dfs[country], country)
        st.pyplot(fig)

    # Wind Rose Plot
    if st.sidebar.checkbox("Show Wind Rose Plot"):
        st.subheader("Wind Rose Plot")
        fig=plot_wind_rose(cleaned_dfs[country], country)
        st.pyplot(fig)

    # Histograms
    if st.sidebar.checkbox("Show Histograms"):
        st.subheader("Histograms")
        columns = st.multiselect("Select columns for Histograms", 
                                 cleaned_dfs[country].columns)
        if columns:
            fig=plot_histograms(cleaned_dfs[country], columns, country)
            st.pyplot(fig)


    # Bubble Chart
    if st.sidebar.checkbox("Show Bubble Chart"):
        st.subheader("Bubble Chart")
        fig=plot_bubble_chart(cleaned_dfs[country], country)
        st.pyplot(fig)

    # Humidity Impact Analysis
    if st.sidebar.checkbox("Show Humidity Impact Analysis"):
        st.subheader("Humidity Impact Analysis")
        fig=analyze_humidity_impact({country: cleaned_dfs[country]})
        st.pyplot(fig)

if __name__ == "__main__":
    main()