import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

def Chart_2():
    # Streamlit application title
    st.title("Linear Regression: Store Area vs. Daily Customer Count")

    # Run the function and get results
    #mse, r2, coef, intercept, fig = train_model('Stores.csv')
    # Load data
    df = pd.read_csv('data/Stores.csv')

    # Select columns of interest
    X = df[['Store_Area']]  # Independent variable
    y = df['Daily_Customer_Count']  # Dependent variable

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict values for the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_test, y_test, color='blue', label='Actual Data')  # Actual data
    ax.plot(X_test, y_pred, color='red', label='Regression Line')  # Regression line
    ax.set_title('Relationship Between Store Area and Daily Customer Count')
    ax.set_xlabel('Store Area')
    ax.set_ylabel('Daily Customer Count')
    ax.legend()
    ax.grid(True)

    # Display metrics in the app
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Coefficient of Determination (R²):** {r2:.2f}")
    st.write(f"**Regression Line Equation:** y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

    # Show the figure in Streamlit
    st.plotly_chart(fig)
