import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def Chart_7():
    st.title("Store Sales Trend Over Time")
    # Application title
    st.markdown("<h2>Time Series Analysis</h2>",
    unsafe_allow_html=True)
    
    # Load data
    df = pd.read_csv("data/Stores.csv")
    df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

    # Remove outliers 
    Q1 = df['Store_Sales'].quantile(0.25)
    Q3 = df['Store_Sales'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Store_Sales'] >= Q1 - 1.5 * IQR) & (df['Store_Sales'] <= Q3 + 1.5 * IQR)]

    # Reset index and regenerating the date column
    df = df.reset_index(drop=True)
    df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    df['Time_Index'] = np.arange(len(df))
    df.set_index('Date', inplace=True)

    # Calculate the moving average - Rolling mean
    # The moving average smoothens the data, which can help highlight underlying trends by reducing the impact of noise and short-term fluctuations.
    df['Rolling_Mean'] = df['Store_Sales'].rolling(window=7).mean()

    # Scale data for regression
    scaler = StandardScaler()
    df['Store_Sales_Scaled'] = scaler.fit_transform(df[['Store_Sales']])

    # Linear Regression
    X = df[['Time_Index']]
    y = df['Store_Sales_Scaled']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Linear Regression Plot
    st.markdown("---")
    st.subheader("Linear Regression ")
    st.write(f"**R² Score:** `{r2:.4f}`")
    st.write(f"**Mean Squared Error (MSE):** `{mse:.4f}`")


    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(x=df.index, y=df['Store_Sales'], name='Actual Sales', line=dict(color='#92cad1')))
    fig_lr.add_trace(go.Scatter(x=df.index, y=y_pred * scaler.scale_[0] + scaler.mean_[0], name='Linear Trend', line=dict(color='#ff0080')))
    fig_lr.add_trace(go.Scatter(x=df.index, y=df['Rolling_Mean'], name='7-Day Rolling Avg', line=dict(color='#d6d727', dash='dot')))

    fig_lr.update_layout(title='Linear Regression with Rolling Average', xaxis_title='Date', yaxis_title='Scaled Sales', template='plotly_white')
    st.plotly_chart(fig_lr, use_container_width=True)

   
    # Time Series Decomposition
    st.markdown("---")
    result = seasonal_decompose(df['Store_Sales'], model='additive', period=30)

    fig_decomp = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonality", "Residual"),
        vertical_spacing=0.05
    )
    fig_decomp.add_trace(go.Scatter(x=df.index, y=result.observed, name="Observed",line=dict(color='#79ccb3')), row=1, col=1)
    fig_decomp.add_trace(go.Scatter(x=df.index, y=result.trend, name="Trend",line=dict(color='#d6d727')), row=2, col=1)
    fig_decomp.add_trace(go.Scatter(x=df.index, y=result.seasonal, name="Seasonality",line=dict(color='#92cad1')), row=3, col=1)
    fig_decomp.add_trace(go.Scatter(x=df.index, y=result.resid, name="Residual",line=dict(color='#e9724d')), row=4, col=1)

    fig_decomp.update_layout(
        height=800,
        title_text="Time Series Decomposition Components",
        showlegend=False,
        template="plotly_white"
    )
    st.plotly_chart(fig_decomp, use_container_width=True)

   
    # ARIMA Forecast
    st.markdown("---")

    arima_model = ARIMA(df['Store_Sales'], order=(1, 1, 1))
    arima_fitted = arima_model.fit()
    forecast = arima_fitted.forecast(steps=30)
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
    forecast.index = forecast_index

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df.index, y=df['Store_Sales'], name='Historical Sales', line=dict(color='#d6d727')))
    fig_forecast.add_trace(go.Scatter(x=forecast.index, y=forecast, name='Forecast (30 Days)', line=dict(color='#92cad1')))
    fig_forecast.update_layout(title="ARIMA Forecast of Store Sales", xaxis_title="Date", yaxis_title="Sales", template="plotly_white")
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.dataframe(forecast.to_frame(name="Forecasted Sales"))
   
