import streamlit as st
import os
import pandas as pd


from modules.module1 import Chart_1
from modules.module2 import Chart_2
from modules.module3 import Chart_3
from modules.module4 import Chart_4
from modules.module5 import Chart_5
from modules.module6 import Chart_6
from modules.module7 import Chart_7
from modules.module8 import Chart_8


#redirection to the dashboard page
if "page" not in st.session_state:
    st.session_state.page = "main"


if st.session_state.page == "dashboard":
    st.switch_page("pages/dashboard.py")


#code to selecct the visualization
st.sidebar.header("Select a Visualization")
option = st.sidebar.selectbox(
    "Select your chart:",
    ("", 
    "1. Store Area Distribution", 
    "2. Relationship Between Area and Daily Customers",
    "3. Distribution of Available Products",
    "4. Relationship Between Customers and Sales",
    "5. Sales Distribution",
    "6. Average Customers by Area",
    "7. Sales Trend Over Time",
    "8. Correlation Heatmap")
)

# Show charts based on the selected option
if option == "1. Store Area Distribution":
    Chart_1()

elif option == "2. Relationship Between Area and Daily Customers":
    Chart_2()
    
elif option == "3. Distribution of Available Products":
    Chart_3()

elif option == "4. Relationship Between Customers and Sales":
    Chart_4()

elif option == "5. Sales Distribution":
    Chart_5()

elif option == "6. Average Customers by Area":
    Chart_6()

elif option == "7. Sales Trend Over Time":
    Chart_7()

elif option == "8. Correlation Heatmap":
    Chart_8()

else:
    st.write(" ")


st.write("INFO-6151 Data Visualization for Machine Learning")
st.write("Group #: 4-Group Members:")
st.write("- Abdallah Waked")
st.write("- Xavier Merino Mino")  
st.write("- Diego Bernal")  

#buttons
if st.button("Go to Dashboard"):
    st.session_state.page = "dashboard"
    st.rerun()

# GitHub logo
st.markdown(
    """
    <a href="https://github.com/Didark8810/INFO6151-Project-1" target="_blank">
        <button style="background-color:#24292e; color:white; border:none; padding:10px 15px; font-size:16px; border-radius:5px; cursor:pointer;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" style="vertical-align:middle; margin-right:10px;">
            Show Code
        </button>
    </a>
    """,
    unsafe_allow_html=True
)

st.write("Base information about the dataset Stores.csv")

df = pd.read_csv('data/Stores.csv')
st.write(df.head(15))