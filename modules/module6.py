import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import plotly.express as px
import plotly.graph_objs as go

# Function to generate a bar plot
def bar_plot(fig_size, data_x, data_y, title, x_label, y_label, colors=['skyblue'], rotation_x=0):
    # Generate bar plot with Plotly
    fig = px.bar(x=data_x, y=data_y, color_discrete_sequence=colors, title=title)
    
    # Update layout
    fig.update_layout(
        title={'text': title, 'font': {'size': 15, 'weight': 'bold'}},
        xaxis_title={'text': x_label, 'font': {'size': 15, 'weight': 'bold'}},
        yaxis_title={'text': y_label, 'font': {'size': 15, 'weight': 'bold'}},
        width=fig_size[0] * 100,  # Convert figsize to pixels
        height=fig_size[1] * 100,
        showlegend=False,
        template='plotly_white',
        plot_bgcolor='white'
    )
    
    # Rotate X-axis labels
    fig.update_xaxes(tickangle=rotation_x)
    
    return fig

def Chart_6():

    # Application title
    st.title("CH6-Average Daily Customer Count by Store Area")

    # Load data
    df = pd.read_csv('data/Stores.csv')

    # Define store area ranges
    n_bins = st.slider("Select the number of ranges:", min_value=3, max_value=10, value=5)
    binning = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

    # Apply discretization to create store area ranges
    df['Store_Area_Range'] = binning.fit_transform(df[['Store_Area']]).astype(int)

    # Calculate the average Daily_Customer_Count by store area range
    average_customers = df.groupby('Store_Area_Range')['Daily_Customer_Count'].mean()

    # Create labels for the ranges
    bin_edges = binning.bin_edges_[0]
    labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(n_bins)]

    # Visualize the average daily customers by store area range
    #st.subheader("Average Daily Customers by Store Area Range")

    fig = bar_plot((8, 6), labels, average_customers, 
                   'Average daily customer per range', 
                   'Store Area Range', 'Daily Customer Average', ['skyblue'], rotation_x = 45)
    
    st.plotly_chart(fig)
