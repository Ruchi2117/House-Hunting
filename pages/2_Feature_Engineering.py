import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

st.title("⚙️ Feature Engineering")

# Sidebar configuration
st.sidebar.image(
    "https://img.icons8.com/clouds/200/000000/home.png",
    width=80
)
st.sidebar.title(" Navigation")
st.sidebar.markdown("---")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/house_data.csv")

df = load_data()

st.write("### 📋 Original Data")
st.write(f"Shape: {df.shape}")

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove target variable from numerical columns
if 'price' in numerical_cols:
    numerical_cols.remove('price')

st.write("#### 🔍 Columns Overview")
col1, col2 = st.columns(2)
with col1:
    st.write("**Numerical Features:**", ", ".join(numerical_cols))
with col2:
    st.write("**Categorical Features:**", ", ".join(categorical_cols))

# Feature Engineering Options
st.write("### 🛠 Feature Engineering Options")

# 1. Handle Categorical Variables
st.write("#### 🔤 Categorical Encoding")
encode_method = st.radio(
    "Select encoding method for categorical variables:",
    ["One-Hot Encoding", "Ordinal Encoding"],
    index=0
)

# 2. Feature Scaling
st.write("#### ⚖️ Feature Scaling")
scale_method = st.radio(
    "Select scaling method for numerical features:",
    ["Standard Scaler", "MinMax Scaler", "None"],
    index=0
)

# 3. Feature Creation
st.write("#### ✨ Feature Creation")
create_price_per_sqft = st.checkbox("Create 'Price per sq.ft' feature", value=True)

# Apply Feature Engineering
if st.button("🔧 Apply Feature Engineering"):
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # 1. Create new features
    if create_price_per_sqft and 'area' in df_processed.columns and 'price' in df_processed.columns:
        df_processed['price_per_sqft'] = df_processed['price'] / df_processed['area']
        numerical_cols.append('price_per_sqft')
    
    # 2. Create transformers
    numerical_transformer = StandardScaler() if scale_method == "Standard Scaler" else \
                          (MinMaxScaler() if scale_method == "MinMax Scaler" else 'passthrough')
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) \
                              if encode_method == "One-Hot Encoding" else 'passthrough'
    
    # 3. Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # 4. Apply transformations
    X = df_processed.drop('price', axis=1) if 'price' in df_processed.columns else df_processed
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    if encode_method == "One-Hot Encoding" and len(categorical_cols) > 0:
        ohe = preprocessor.named_transformers_['cat']
        ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
        all_feature_names = numerical_cols + ohe_feature_names.tolist()
    else:
        all_feature_names = numerical_cols + categorical_cols
    
    # Convert back to DataFrame
    if hasattr(X_processed, 'toarray'):  # If sparse matrix
        X_processed = X_processed.toarray()
    
    df_processed = pd.DataFrame(X_processed, columns=all_feature_names)
    
    # Add back the target variable if it exists
    if 'price' in df.columns:
        df_processed['price'] = df['price'].values
    
    # Save processed data
    os.makedirs('data', exist_ok=True)
    df_processed.to_csv("data/processed_house_data.csv", index=False)
    
    # Save the preprocessor
    joblib.dump(preprocessor, 'model/preprocessor.pkl')
    
    st.success("✅ Feature engineering completed successfully!")
    
    # Show processed data
    st.write("### 🔢 Processed Data")
    st.write(f"Shape after processing: {df_processed.shape}")
    st.dataframe(df_processed.head())
    
    # Download button for processed data
    st.download_button(
        label="📥 Download Processed Data",
        data=df_processed.to_csv(index=False),
        file_name='processed_house_data.csv',
        mime='text/csv',
    )
else:
    st.info("ℹ️ Configure the options above and click 'Apply Feature Engineering' to process the data.")

# Add some space at the bottom
st.write("")
st.write("")
st.write("*Note: After processing, proceed to the 'Model Training' page to train your model.*")
