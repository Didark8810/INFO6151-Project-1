import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def Chart_8():
    st.title("Correlation Matrix Heatmap")
        
    # Load data
    df = pd.read_csv("data/Stores.csv")

    # Select relevant numeric columns
    selected_cols = ['Store_Area', 'Items_Available', 'Daily_Customer_Count', 'Store_Sales']
    
    # Calculate correlation matrix
    corr_matrix = df[selected_cols].corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Store Features", fontsize=14)
    st.pyplot(fig)
   
    # Display the chart in Streamlit
    st.pyplot(fig)
