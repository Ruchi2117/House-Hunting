import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("🔍 Exploratory Data Analysis (EDA)")

# Sidebar configuration
st.sidebar.image(
    "https://img.icons8.com/clouds/200/000000/home.png",
    width=80
)
st.sidebar.title(" Navigation")
st.sidebar.markdown("---")

# Create data directory if it doesn't exist
import os
os.makedirs("data", exist_ok=True)

# Load or download data
@st.cache_data
def load_data():
    data_path = "data/house_data.csv"
    
    # If data file doesn't exist, download it
    if not os.path.exists(data_path):
        import requests
        from io import StringIO
        
        # URL to the raw dataset (you can replace this with your dataset URL)
        dataset_url = "https://raw.githubusercontent.com/Ruchi2117/House-Hunting/main/data/house_data.csv"
        
        try:
            response = requests.get(dataset_url)
            response.raise_for_status()
            
            # Save the data locally
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            st.success("✅ Data downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading data: {str(e)}")
            return None
    
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Load the data
df = load_data()

if df is None:
    st.warning("⚠️ No data available. Please check your internet connection or data source.")

# Basic Info
st.write("### 📌 Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Rows", df.shape[0])
    st.metric("Total Columns", df.shape[1])
with col2:
    st.metric("Average Price", f"${df['price'].mean():,.0f}")
    st.metric("Average Area", f"{df['area'].mean():.0f} sq.ft")

# Display first few rows
st.write("### 📋 First 5 Rows")
st.dataframe(df.head())

# Data Types and Missing Values
st.write("### 🔍 Data Types & Missing Values")
st.write(pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isnull().sum(),
    'Unique Values': df.nunique()
}))

# Numerical Features Analysis
st.write("### 📊 Numerical Features Analysis")
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Summary Statistics
st.write("#### 📈 Summary Statistics")
st.dataframe(df[num_cols].describe().T)

# Distribution of Numerical Features
st.write("#### 📊 Distributions")
selected_num_col = st.selectbox("Select a numerical feature:", num_cols)
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df[selected_num_col], kde=True, ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.write("### 🔗 Correlation Heatmap")
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Categorical Features Analysis
st.write("### 📊 Categorical Features Analysis")
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

if cat_cols:
    selected_cat_col = st.selectbox("Select a categorical feature:", cat_cols)
    
    # Value counts
    st.write(f"#### Value Counts for {selected_cat_col}")
    value_counts = df[selected_cat_col].value_counts()
    st.bar_chart(value_counts)
    
    # Box plot for categorical vs price
    if len(cat_cols) > 0:
        st.write(f"#### {selected_cat_col} vs Price")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=selected_cat_col, y='price', data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Pairplot for numerical features (sample for speed)
if st.checkbox("Show Pairplot (First 100 samples)"):
    st.write("### 🔄 Pairplot of Numerical Features (First 100 samples)")
    pairplot_fig = sns.pairplot(df[num_cols].head(100), corner=True)
    st.pyplot(pairplot_fig)
