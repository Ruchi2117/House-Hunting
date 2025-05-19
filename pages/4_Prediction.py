import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏡 House Price Prediction")

# Sidebar configuration
st.sidebar.image(
    "https://img.icons8.com/clouds/200/000000/home.png",
    width=80
)
st.sidebar.title(" Navigation")
st.sidebar.markdown("---")

# Load Model with error handling
@st.cache_resource
def load_model():
    try:
        model_path = "model/house_price_model.pkl"
        if not os.path.exists(model_path):
            st.error("❌ Model not found. Please train a model first using the Model Training page.")
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_model()

if model is not None:
    # Create input form
    with st.form("prediction_form"):
        st.header("📝 Enter Property Details")
        
        # Layout in columns for better organization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Basic Information")
            area = st.number_input("Area (sq.ft)", min_value=300, max_value=15000, step=50, value=1000)
            bedrooms = st.slider("Bedrooms", 1, 10, 2)
            bathrooms = st.slider("Bathrooms", 1, 10, 2)
            stories = st.slider("Number of Stories", 1, 4, 1)
            
        with col2:
            st.subheader("Location")
            mainroad = st.selectbox("Main Road Access", ["Yes", "No"])
            guestroom = st.selectbox("Guest Room", ["Yes", "No"])
            basement = st.selectbox("Basement", ["Yes", "No"])
            hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
            
        with col3:
            st.subheader("Additional Features")
            airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
            parking = st.slider("Parking Spaces", 0, 3, 0)
            prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
            furnishingstatus = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
        
        # Submit button
        submitted = st.form_submit_button("🚀 Predict Price")
        
        if submitted:
            with st.spinner("Predicting..."):
                # Prepare input data with one-hot encoding for all categorical variables
                input_data = {
                    'area': [area],
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'stories': [stories],
                    'parking': [parking],
                    'price_per_sqft': [area / 1000],  # Assuming this was a feature in training
                    'mainroad_no': [1 if mainroad == "No" else 0],
                    'mainroad_yes': [1 if mainroad == "Yes" else 0],
                    'guestroom_no': [1 if guestroom == "No" else 0],
                    'guestroom_yes': [1 if guestroom == "Yes" else 0],
                    'basement_no': [1 if basement == "No" else 0],
                    'basement_yes': [1 if basement == "Yes" else 0],
                    'hotwaterheating_no': [1 if hotwaterheating == "No" else 0],
                    'hotwaterheating_yes': [1 if hotwaterheating == "Yes" else 0],
                    'airconditioning_no': [1 if airconditioning == "No" else 0],
                    'airconditioning_yes': [1 if airconditioning == "Yes" else 0],
                    'prefarea_no': [1 if prefarea == "No" else 0],
                    'prefarea_yes': [1 if prefarea == "Yes" else 0],
                    'furnishingstatus_furnished': [1 if furnishingstatus.lower() == 'furnished' else 0],
                    'furnishingstatus_semi-furnished': [1 if furnishingstatus.lower() == 'semi-furnished' else 0],
                    'furnishingstatus_unfurnished': [1 if furnishingstatus.lower() == 'unfurnished' else 0]
                }
                
                # Create DataFrame
                input_df = pd.DataFrame(input_data)
                
                # Expected columns based on the training data
                expected_columns = [
                    'area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price_per_sqft',
                    'mainroad_no', 'mainroad_yes', 'guestroom_no', 'guestroom_yes',
                    'basement_no', 'basement_yes', 'hotwaterheating_no', 'hotwaterheating_yes',
                    'airconditioning_no', 'airconditioning_yes', 'prefarea_no', 'prefarea_yes',
                    'furnishingstatus_furnished', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
                ]
                
                # Ensure all expected columns are present (in case the model expects them in a specific order)
                input_df = input_df.reindex(columns=expected_columns, fill_value=0)
                
                # Make prediction
                try:
                    predicted_price = model.predict(input_df)[0]
                    
                    # Display result
                    st.success("### 🎉 Prediction Complete!")
                    
                    # Show prediction with nice formatting
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Estimated Property Value", f"${predicted_price:,.2f}")
                        
                        # Add some visual feedback
                        if predicted_price > 1000000:
                            st.info("💎 Premium property detected!")
                        elif predicted_price > 500000:
                            st.info("🏡 Great value for money!")
                        else:
                            st.info("💰 Affordable option!")
                    
                    with col2:
                        # Show a simple gauge chart
                        import plotly.graph_objects as go
                        
                        # Simple price range estimation (adjust based on your data)
                        min_price = 100000  # Adjust based on your data
                        max_price = 5000000  # Adjust based on your data
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=predicted_price,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Price Estimation"},
                            delta={'reference': (min_price + max_price)/2, 'increasing': {'symbol': '▲'}},
                            gauge={
                                'axis': {'range': [min_price, max_price]},
                                'bar': {'color': "#1f77b4"},
                                'steps': [
                                    {'range': [min_price, min_price + (max_price-min_price)/3], 'color': "lightgray"},
                                    {'range': [min_price + (max_price-min_price)/3, min_price + 2*(max_price-min_price)/3], 'color': "darkgray"},
                                    {'range': [min_price + 2*(max_price-min_price)/3, max_price], 'color': "gray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': predicted_price
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("📊 What's driving this prediction?")
                        feature_importance = pd.DataFrame({
                            'Feature': input_df.columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        # Only show top 5 features
                        top_features = feature_importance.head(5)
                        
                        # Create a horizontal bar chart
                        fig, ax = plt.subplots()
                        ax.barh(top_features['Feature'], top_features['Importance'])
                        ax.set_xlabel('Importance')
                        ax.set_title('Top 5 Influential Features')
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.write("Input DataFrame columns:", input_df.columns.tolist())
                    st.write("Input DataFrame values:", input_df.values.tolist())
    
    # Add some helpful information
    st.markdown("---")
    st.info("💡 Tip: For best results, ensure all fields are filled accurately. The prediction is based on the trained model's learning from historical data.")
    
    # Add a button to go to model training page
    if st.button("🔄 Train a New Model"):
        st.switch_page("pages/3_Model_Training.py")
