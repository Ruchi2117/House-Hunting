import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Project Summary", layout="wide")
st.title("🏠 House Price Prediction - Project Summary")

# Sidebar configuration
st.sidebar.image(
    "https://img.icons8.com/clouds/200/000000/home.png",
    width=80
)
st.sidebar.title(" Navigation")
st.sidebar.markdown("---")

# Project Overview
st.header("📌 Project Overview")
st.markdown("""
This project is an end-to-end machine learning application for predicting house prices based on various features. 
The application is built using Streamlit and includes the following components:

1. **Exploratory Data Analysis (EDA)** - Understand the dataset and visualize key patterns
2. **Feature Engineering** - Preprocess and transform the data for modeling
3. **Model Training** - Train and evaluate multiple machine learning models
4. **Prediction** - Make predictions on new data using the trained model
""")

# Dataset Information
try:
    data_path = "data/processed_house_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.header("📊 Dataset Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Numeric Columns", len(df.select_dtypes(include=['int64', 'float64']).columns))
        
        st.subheader("📋 Sample Data")
        st.dataframe(df.head())
        
        st.subheader("📈 Data Types")
        st.dataframe(df.dtypes.rename('Data Type'))
        
except Exception as e:
    st.warning("⚠️ Processed data not found. Please run the Feature Engineering page first.")

# Project Structure
st.header("📁 Project Structure")
st.code("""📦 streamlit-house-price-prediction-main
├── 📂 data/
│   ├── house_data.csv          # Raw dataset
│   ├── processed_house_data.csv # Processed dataset
│   └── house_price_model.pkl    # Legacy model (duplicate)
├── 📂 model/
│   ├── house_price_model.pkl    # Trained model
│   └── scaler.pkl              # Feature scaler
├── 📂 pages/
│   ├── 0_Summary.py           # This summary page
│   ├── 1_EDA.py               # Exploratory Data Analysis
│   ├── 2_Feature_Engineering.py # Data preprocessing
│   ├── 3_Model_Training.py     # Model training and evaluation
│   └── 4_Prediction.py         # Make predictions
├── 📄 README.md                # Project documentation
└── 📄 requirements.txt         # Dependencies
""", language="bash")

# How to Use
st.header("🚀 How to Use")
st.markdown("""
1. **Start with EDA** - Explore the dataset and understand the features
2. **Feature Engineering** - Preprocess the data and handle missing values
3. **Model Training** - Train and evaluate different models
4. **Prediction** - Use the trained model to make predictions

Navigate through the pages using the sidebar to access each section.
""")

# Dependencies
st.header("📦 Dependencies")
st.code("""
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
joblib
plotly
""", language="bash")

# Note
st.info("💡 Note: Make sure to run the pages in order (1 → 2 → 3 → 4) for the first time to ensure all dependencies are properly set up.")
