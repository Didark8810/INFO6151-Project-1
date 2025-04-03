import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

def Chart_3():
    
    st.title("Distribution of Items Available ")

    
    df = pd.read_csv('data/Stores.csv')

    # items ranges
    n_bins = st.slider("Select the number of clusters", min_value=2, max_value=20, value=5)
    binning = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

    # Create ranges
    df['Items_Range'] = binning.fit_transform(df[['Items_Available']]).astype(int)

    # Count the number of stores in each range
    items_distribution = df['Items_Range'].value_counts().sort_index()

    # Create labels 
    bin_edges = binning.bin_edges_[0]
    labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(n_bins)]

    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, items_distribution, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of Available Items by Range')
    ax.set_xlabel('Available Items Range')
    ax.set_ylabel('Number of Stores')
    ax.set_xticklabels(labels, rotation=45)  # Rotate X-axis labels
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    
    st.pyplot(fig)
