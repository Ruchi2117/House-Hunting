import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

st.title("📊 Model Training")

# Load data with error handling
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/processed_house_data.csv")
    except FileNotFoundError:
        st.error("❌ Processed data not found. Please run the Feature Engineering page first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

df = load_data()

if df is not None:
    st.write("### 📋 Processed Data Overview")
    st.write(f"Shape: {df.shape}")
    
    # Check if target variable exists
    if 'price' not in df.columns:
        st.error("❌ 'price' column not found in the dataset. Please check your data processing.")
    else:
        # Sidebar for model configuration
        st.sidebar.header("Model Configuration")
        
        # Model selection
        model_name = st.sidebar.selectbox(
            "Select Model",
            ["Linear Regression", "Ridge", "Lasso", "ElasticNet", 
             "Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"]
        )
        
        # Common parameters
        test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.sidebar.number_input("Random State", 0, 100, 42)
        
        # Model-specific parameters
        if model_name in ["Ridge", "Lasso", "ElasticNet"]:
            alpha = st.sidebar.number_input("Alpha", 0.01, 10.0, 1.0, 0.1)
        
        if model_name == "Random Forest":
            n_estimators = st.sidebar.number_input("Number of Trees", 10, 500, 100, 10)
            max_depth = st.sidebar.number_input("Max Depth", 1, 20, 10, 1)
        
        if model_name in ["Gradient Boosting", "XGBoost"]:
            n_estimators = st.sidebar.number_input("Number of Estimators", 10, 500, 100, 10)
            learning_rate = st.sidebar.number_input("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        
        # Prepare data
        X = df.drop(columns=["price"])
        y = df["price"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Initialize and train model
        model = None
        
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Ridge":
            model = Ridge(alpha=alpha, random_state=random_state)
        elif model_name == "Lasso":
            model = Lasso(alpha=alpha, random_state=random_state)
        elif model_name == "ElasticNet":
            model = ElasticNet(alpha=alpha, random_state=random_state)
        elif model_name == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_name == "Gradient Boosting":
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=5,  # Default value
                random_state=random_state
            )
        elif model_name == "XGBoost":
            model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=6,  # XGBoost default
                random_state=random_state,
                n_jobs=-1
            )
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor(
                max_depth=5,  # Prevent overfitting
                random_state=random_state
            )
        
        # Train model
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                # Fit model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                # Save model
                os.makedirs('model', exist_ok=True)
                model_path = "model/house_price_model.pkl"
                joblib.dump(model, model_path)
                
                # Display results
                st.success("✅ Model trained and saved successfully!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train R²", f"{train_r2:.4f}")
                    st.metric("Train RMSE", f"{train_rmse:,.2f}")
                with col2:
                    st.metric("Test R²", f"{test_r2:.4f}")
                    st.metric("Test RMSE", f"{test_rmse:,.2f}")
                with col3:
                    st.metric("CV R² (Mean ± Std)", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                # Actual vs Predicted plot
                st.write("### 📈 Actual vs Predicted Prices")
                fig, ax = plt.subplots(figsize=(10, 6))
                max_val = max(y_test.max(), y_test_pred.max()) * 1.1
                ax.scatter(y_test, y_test_pred, alpha=0.5)
                ax.plot([0, max_val], [0, max_val], 'r--')
                ax.set_xlabel("Actual Prices")
                ax.set_ylabel("Predicted Prices")
                ax.set_title("Actual vs Predicted House Prices")
                st.pyplot(fig)
                
                # Feature importance if available
                if hasattr(model, 'feature_importances_'):
                    st.write("### 📊 Feature Importance")
                    importances = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='importance', y='feature', data=importances, ax=ax)
                    ax.set_title("Top 10 Most Important Features")
                    st.pyplot(fig)
                
                # Download model button
                with open(model_path, 'rb') as f:
                    st.download_button(
                        label="💾 Download Model",
                        data=f,
                        file_name="house_price_model.pkl",
                        mime="application/octet-stream"
                    )
