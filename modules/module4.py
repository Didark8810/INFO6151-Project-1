import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
import plotly.express as px

# Funtion to create a scatter plot
def scatter_plot(fig_size, data, x, y, colors, title, x_label, y_label, grid=True, rows=100):
    # Validations of max value
    max_rows = data[x].shape[0]
    
    if rows <= 0:
        rows = 1
    elif rows >= max_rows:
        rows = max_rows
    
    # Plotting with Plotly
    fig = px.scatter(data.iloc[:rows], x=x, y=y, color_discrete_sequence=[colors[0]])
    
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
    
    if grid:
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
    else:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    
    return fig

# Funtion to generate a logistic regression model
def logistic_regression(X_train_dcc, X_test_dcc, y_train_ss, y_test_ss, n = 2):
    poly = PolynomialFeatures(degree=n)
    X_poly_train = poly.fit_transform(X_train_dcc)
    X_poly_test = poly.transform(X_test_dcc)
    
    lasso_model = Lasso(alpha=0.001, max_iter=50 ** n)
    # Train the model with training data
    lasso_model.fit(X_poly_train, y_train_ss)
    
    # Make predictions with the validation dataset
    y_val_pred_lasso = lasso_model.predict(X_poly_test)
    
    # Calculate the MSE to check the model with the validation dataset
    mse_val_lasso = mean_squared_error(y_test_ss, y_val_pred_lasso)

    return lasso_model, y_val_pred_lasso, mse_val_lasso

# Funtion to generate a neural network
def neural_network(hidden_layer_sizes, max_iter, activation, X_train, y_train, X_test, y_test):
    # Initialize the Neural Network model
    mlp_model = MLPRegressor(activation=activation,
                            hidden_layer_sizes=(hidden_layer_sizes),
                            alpha=0.001,
                            random_state=20,
                            max_iter = max_iter,
                            early_stopping=False)
    
    # Fit the model to the training data using GridSearchCV
    mlp_model.fit(X_train, y_train)
        
    # Make predictions on the test set using the best model
    y_pred = mlp_model.predict(X_test)
        
    return mean_squared_error(y_test, y_pred), y_pred, mlp_model

def Chart_4():
    # Application title
    st.title("CH4-Relationship between Daily Customer Count and Store Sales")

    # Load data
    df = pd.read_csv('data/Stores.csv')

    # --- Preprocessing section ---
    # Daily_Customer_Count and Store_Sales has different ranges.
    # It is necessary to scale them
    scaler = StandardScaler()
    
    # Fit and transform the data
    df_scl = scaler.fit_transform(df)
    #df_scl = scaler.fit_transform(df.iloc[:, :-1])
    
    # Convert standardized data to a DataFrame
    column_names = list(df.columns)
    #column_names = list(list(df.columns)[:-1])
    df_scl = pd.DataFrame(df_scl, columns=column_names)

    # --- Divide the dataset ---
    X_dcc = pd.DataFrame(df_scl.loc[:, 'Daily_Customer_Count'])
    y_ss = pd.DataFrame(df_scl.loc[:, 'Store_Sales'])
    # Split the data into training (80%) and testing sets
    X_train_dcc, X_test_dcc, y_train_ss, y_test_ss = train_test_split(X_dcc, y_ss, test_size=0.2, random_state=42)

    # --- Splitting the variables to analyze them ---
    df_ch4 = df_scl.loc[:, ['Daily_Customer_Count', 'Store_Sales']]
    # Calculate correlation matrix
    corr_matrix_ch4 = df_ch4.corr()
    
    # --- Display metrics in Streamlit ---
    st.subheader("Evaluation:")
    st.write(f"🔹 **Correlation:** {corr_matrix_ch4.iloc[1,0]:.4f}")

    st.subheader("Relationship plot:")
    # Input a slider to select the number of datapoints
    datapoints = st.slider("Select the number of datapoints", min_value=1, 
                           max_value=df_ch4['Daily_Customer_Count'].shape[0], 
                           value=100, key=1)
    
    fig = scatter_plot((8,6), df, 'Daily_Customer_Count', 'Store_Sales', ['blue'], 
                         'Daily customer vs Store sales', 'Daily customer count', 'Store sales', rows = datapoints)

    # --- Display the chart in Streamlit ---
    st.plotly_chart(fig)

    # --- Machine Learning model implementation ---
    st.subheader("Machine Learning Models:")

    option = st.selectbox(
        "Select your model:",
        ("", 
        "1. Logistic Regression", 
        "2. Neural Network")
    )
    
    # process models
    if option == "1. Logistic Regression":
        option = st.selectbox(
            "Select the grade of the model:",
            ("", 
            "Grade 2", 
            "Grade 3")
        )

        grade = 0
        # Selection the grade of the model
        if option == "Grade 2":
            grade = 2
        elif option == "Grade 3":
            grade = 3

        if grade != 0:
            # processing the model
            model, y_pred, mse = logistic_regression(X_train_dcc, X_test_dcc, y_train_ss, y_test_ss, grade)
            
            # --- Display metrics in Streamlit ---
            st.subheader("Evaluation:")
            st.write(f"🔹 **MSE for grade {grade} :** {mse:.4f}")
            
            data = {
                'Real': y_test_ss.iloc[:,0],
                'Predicted': y_pred
            }
            
            # creating a Dataframe object 
            df_pred = pd.DataFrame(data)
            
            # --- Display plot prediction vs real values ---
            # Input a slider to select the number of datapoints
            datapoints = st.slider("Select the number of datapoints", min_value=1, 
                                   max_value=df_ch4['Daily_Customer_Count'].shape[0], 
                                   value=100, key=2)
    
            fig = scatter_plot((8,6), df_pred, 'Real', 'Predicted', ['orange'], 
                                'Real vs Predicted values', 'Real', 'Predicted', rows=datapoints)
    
            # --- Display the chart in Streamlit ---
            st.plotly_chart(fig)
    
    if option == "2. Neural Network":
        mse_nn, y_pred, model_nn = neural_network(40, 4000, 'relu', 
                                        X_train_dcc, y_train_ss, X_test_dcc, y_test_ss)

        # --- Display metrics in Streamlit ---
        st.subheader("Evaluation:")
        st.write(f"🔹 **MSE:** {mse_nn:.4f}")

        data = {
            'Real': y_test_ss.iloc[:,0],
            'Predicted': y_pred
        }
        
        # creating a Dataframe object 
        df_pred = pd.DataFrame(data)
        
        # --- Display plot prediction vs real values ---
        # Input a slider to select the number of datapoints
        datapoints = st.slider("Select the number of datapoints", min_value=1, 
                               max_value=df_ch4['Daily_Customer_Count'].shape[0], 
                               value=100, key=3)

        fig = scatter_plot((8,6), df_pred, 'Real', 'Predicted', ['orange'], 
                            'Real vs Predicted values', 'Real', 'Predicted', rows=datapoints)

        # --- Display the chart in Streamlit ---
        st.plotly_chart(fig)
