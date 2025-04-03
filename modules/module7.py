import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

def Chart_7():
    st.title("CH7-Store Sales Trend Over Time")

    # Load Data
    df = pd.read_csv("data/Stores.csv")
    df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    df['Year'] = df['Date'].dt.year

    # Year filter
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, "All Years")
    selected_year = st.selectbox("Select Year", years)

    if selected_year != "All Years":
        df = df[df['Year'] == selected_year]

    df.set_index('Date', inplace=True)

    # Remove outliers
    Q1 = df['Store_Sales'].quantile(0.25)
    Q3 = df['Store_Sales'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Store_Sales'] >= Q1 - 1.5 * IQR) & (df['Store_Sales'] <= Q3 + 1.5 * IQR)]

    # Prepare data
    df['Time_Index'] = np.arange(len(df))
    df['Rolling_Mean'] = df['Store_Sales'].rolling(window=7).mean()

    # Regression
    scaler = StandardScaler()
    df['Store_Sales_Scaled'] = scaler.fit_transform(df[['Store_Sales']])
    X = df[['Time_Index']]
    y = df['Store_Sales_Scaled']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    total_items = df['Items_Available'].sum()
    total_sales = df['Store_Sales'].sum()

    r2_arrow = "⬆️" if r2 >= 0.75 else "➡️" if r2 >= 0.5 else "⬇️"
    mse_arrow = "⬇️" if mse <= 0.1 else "➡️" if mse <= 0.4 else "⬆️"

    # --- Metrics on Top ---
    st.markdown("---")
    st.markdown("### Model Performance and Summary")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        st.markdown(f"""
        <div style='width:180px; height:150px;border:2px solid #e2e8f0; border-radius:10px; padding:20px; right-margin:10px;text-align:center;'>
            <h5>R² Score {r2_arrow}</h5>
            <h4 style='color:#2b6cb0;'>{r2:.4f}</h4>
        </div>
        """, unsafe_allow_html=True)
    with col_r2:
        st.markdown(f"""
        <div style='width:170px; height:150px;border:2px solid #e2e8f0; border-radius:10px; padding:20px; right-margin:5px;text-align:center;'>
            <h5>MSE {mse_arrow}</h5>
            <h4 style='color:#2b6cb0;'>{mse:.4f}</h4>
        </div>
        """, unsafe_allow_html=True)

    with col_r3:
        st.markdown(f"""
        <div style='width:180px; height:150px;border:2px solid #e2e8f0; border-radius:10px; padding:10px; right-margin:10px;text-align:center;'>
            <h5>Items Available</h5>
            <h4 style='color:#2b6cb0;'>{total_items:,.0f}</h4>
        </div>
        """, unsafe_allow_html=True)



    with col_r4:
        st.markdown(f"""
        <div style='width:170px; height:150px;border:2px solid #e2e8f0; border-radius:10px; padding:10px; right-margin:10px;text-align:center;'>
            <h5>Store Sales</h5>
            <h4 style='color:#2b6cb0;'>${total_sales:,.0f}</h4>
        </div>
        """, unsafe_allow_html=True)

    # --- Linear Regression Plot ---
    st.markdown("#### Linear Regression Trend")
    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(x=df.index, y=df['Store_Sales'], name='Actual Sales', line=dict(color='#92cad1')))
    fig_lr.add_trace(go.Scatter(x=df.index, y=y_pred * scaler.scale_[0] + scaler.mean_[0], name='Linear Trend', line=dict(color='#ff0080')))
    fig_lr.add_trace(go.Scatter(x=df.index, y=df['Rolling_Mean'], name='7-Day Rolling Avg', line=dict(color='#d6d727', dash='dot')))
    fig_lr.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_lr, use_container_width=True)

    # --- Decomposition ---
    if len(df) >= 30:
        result = seasonal_decompose(df['Store_Sales'], model='additive', period=30)
        st.markdown("---")
        st.subheader("Decomposition Components")
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        with row1_col1:
            st.markdown("**Trend**")
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df.index, y=result.trend, name="Trend", line=dict(color='#d6d727')))
            fig_trend.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_trend, use_container_width=True)

        with row1_col2:
            st.markdown("**Seasonality**")
            fig_season = go.Figure()
            fig_season.add_trace(go.Scatter(x=df.index, y=result.seasonal, name="Seasonality", line=dict(color='#92cad1')))
            fig_season.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_season, use_container_width=True)

        with row2_col1:
            st.markdown("**Residual**")
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Scatter(x=df.index, y=result.resid, name="Residual", line=dict(color='#e9724d')))
            fig_resid.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_resid, use_container_width=True)

        with row2_col2:
            st.markdown("**Observed Series**")
            fig_obs = go.Figure()
            fig_obs.add_trace(go.Scatter(x=df.index, y=result.observed, name="Observed", line=dict(color='#79ccb3')))
            fig_obs.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_obs, use_container_width=True)
    else:
        st.warning("Not enough data points for decomposition. Minimum required: 30")

    # --- ARIMA Forecast ---
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

    formatted_forecast = forecast.apply(lambda x: f"{x:,.2f}").to_frame(name="Forecasted Sales")
    formatted_forecast.index = formatted_forecast.index.strftime('%Y-%m-%d')

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig_forecast, use_container_width=True)
    with col2:
        st.markdown("#### Forecasted Sales (Next 30 Days)")
        st.dataframe(formatted_forecast, height=500)
