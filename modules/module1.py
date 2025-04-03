import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def Chart_1():

    
    st.title("CH1-Distribution of Store Area")
    df = pd.read_csv('data/Stores.csv')
    store_areas = df[['Store_Area']]

    
    n_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=3)

    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df['Cluster'] = kmeans.fit_predict(store_areas)
    
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.subheader("Number of Stores in Each Cluster::")
    st.write(cluster_counts)


    # one
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    scatter = ax2.scatter(df['Store_Area'], df['Cluster'], c=df['Cluster'], cmap='viridis', alpha=0.7)
    ax2.set_title('Store Area Clusters')
    ax2.set_xlabel('Store Area')
    ax2.set_ylabel('Cluster')
    fig2.colorbar(scatter, label='Cluster')
    ax2.grid(True)

    st.pyplot(fig2)
    
    # two
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(df['Store_Area'], bins=30, edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution of Store Areas')
    ax1.set_xlabel('Store Area')
    ax1.set_ylabel('Frequency')
    ax1.grid(True)
    
    st.plotly_chart(fig1)

 
