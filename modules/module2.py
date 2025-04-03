import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def Chart_2():
    st.title("Relationship between Store Area and Daily Customer Count")
    
    df = pd.read_csv('data/Stores.csv')

    
    X = df[['Store_Area']]  
    y = df['Daily_Customer_Count']  

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    #  Random Forest
    rf = RandomForestRegressor(random_state=42)

    # GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

   # Display the best parameters found
    st.subheader("Best Parameters:")
    st.write(grid_search.best_params_)

    # Make predictions with the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    
    st.subheader("Model Metrics:")
    st.write(f"Mean Squared Error (MSE):{mse:.2f}")
    st.write(f"R² Coefficient of Determination: {r2:.2f}")

    
    st.subheader("Scatter Plot and Prediction")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_test, y_test, color='blue', label='Actual Data')
    ax.scatter(X_test, y_pred, color='red', label='Predictions')
    ax.set_title('Relationship Between Store Area and Daily Customer Count')
    ax.set_xlabel('Store Area')
    ax.set_ylabel('Daily Customer Count')
    ax.legend()
    ax.grid(True)


    
    st.plotly_chart(fig)


Chart_2()