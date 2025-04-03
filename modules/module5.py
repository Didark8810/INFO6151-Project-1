import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from scipy.stats import norm

def histogram(fig_size, data_x, x_label, y_label, title, pdf=False, color='blue', bins=10):
    # Generate histogram
    hist_data = [data_x]
    group_labels = ['Histogram']
    
    fig = ff.create_distplot(hist_data, group_labels, bin_size=(max(data_x) - min(data_x)) / bins, 
                             show_hist=True, show_rug=False, colors=[color])
    
    if pdf:
        # Fit a normal distribution to the data
        mu, std = norm.fit(data_x)
        
        # Plot the PDF line
        x = np.linspace(min(data_x), max(data_x), 100)
        p = norm.pdf(x, mu, std)
        pdf_trace = go.Scatter(x=x, y=p, mode='lines', name='PDF', line=dict(color='black', width=2))
        fig.add_trace(pdf_trace)
    
    # Update layout
    fig.update_layout(
        title={'text': title, 'font': {'size': 15, 'weight': 'bold'}},
        xaxis_title={'text': x_label, 'font': {'size': 15, 'weight': 'bold'}},
        yaxis_title={'text': y_label, 'font': {'size': 15, 'weight': 'bold'}},
        width=fig_size[0] * 100,  # Convert figsize to pixels
        height=fig_size[1] * 100,
        showlegend=True,
        template='plotly_white',
        plot_bgcolor='white'
    )
    
    return fig

def box_plot(fig_size, data_x, data_y, title, x_label, y_label, linewidth=1, marker='o', markersize=1):
    # If there is no data_y, a simple boxplot is shown
    if data_y is None:
        fig = px.box(data_x, points='all', title=title)
    else:
        fig = px.box(data_x, y=data_y, points='all', title=title)
    
    # Update layout
    fig.update_layout(
        title={'text': title, 'font': {'size': 15}},
        xaxis_title={'text': x_label, 'font': {'size': 15}},
        yaxis_title={'text': y_label, 'font': {'size': 15}},
        width=fig_size[0] * 100,  # Convert figsize to pixels
        height=fig_size[1] * 100,
        showlegend=False,
        template='plotly_white',
        plot_bgcolor='white'
    )
    
    return fig

def Chart_5():

    # Application title
    st.title("CH5-Store Sales Distribution Analysis")

    # Load data
    df = pd.read_csv('data/Stores.csv')

    # Analyze the distribution of Store_Sales
    store_sales = df['Store_Sales']

    # Calculate descriptive statistics
    mean_sales = store_sales.mean()
    median_sales = store_sales.median()
    q1 = store_sales.quantile(0.25)  # First quartile (25%)
    q3 = store_sales.quantile(0.75)  # Third quartile (75%)
    iqr = q3 - q1  # Interquartile range (IQR)

    # Identify outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = store_sales[(store_sales < lower_bound) | (store_sales > upper_bound)]

    # Visualize the distribution
    st.subheader("Distribution of Store Sales")

    fig = histogram((8, 6), store_sales, 'Store_Sales', 'Density', 'Store Sales histogram distribution', 
                      True, color = 'orange')

    st.plotly_chart(fig)

    # Display statistics in Streamlit
    st.subheader("Descriptive Statistics")
    st.write(f"🔹 **Mean Sales:** {mean_sales:.2f}")
    st.write(f"🔹 **Median Sales:** {median_sales:.2f}")
    st.write(f"🔹 **First Quartile (Q1):** {q1:.2f}")
    st.write(f"🔹 **Third Quartile (Q3):** {q3:.2f}")
    st.write(f"🔹 **Interquartile Range (IQR):** {iqr:.2f}")
    st.write(f"🔹 **Number of Outliers Detected:** {len(outliers)}")

    # Display Box Plot
    st.subheader("Box Plot")

    fig = box_plot((8, 6), store_sales, None, 'Store sales boxplot distribution', 
                   'Store sales distribution customer', '')
    st.plotly_chart(fig)
