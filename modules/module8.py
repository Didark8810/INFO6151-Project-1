import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

def Chart_8():
    st.title("Correlation Matrix Heatmap")

    # Load data
    df = pd.read_csv("data/Stores.csv")

    # Select numeric columns
    selected_cols = ['Store_Area', 'Items_Available', 'Daily_Customer_Count', 'Store_Sales']

    # Variable selector
    selected_vars = st.multiselect("Select variables for correlation:", selected_cols, default=selected_cols)

    if len(selected_vars) < 2:
        st.warning("Please select at least two variables to display correlation.")
        return

    # Correlation matrix
    corr_matrix = df[selected_vars].corr()

    # View choice
    view = st.radio("Choose view:", ["Heatmap", "Table"], horizontal=True)

    if view == "Heatmap":
        st.subheader("Correlation Heatmap")
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu",
                        labels=dict(color="Correlation"), aspect="auto")
        fig.update_layout(height=600, width=700)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("Correlation Table")

        # Styled correlation table
        def color_corr(val):
            color = 'blue' if val == 1 else 'green' if val > 0 else 'red'
            return f'color: {color}'

        styled_corr = corr_matrix.style.format("{:.2f}").applymap(color_corr)
        st.dataframe(styled_corr)