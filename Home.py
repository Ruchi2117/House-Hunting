import streamlit as st
import base64
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="House Hunting - Your AI Home Buying Assistant",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
local_css("style.css")

# Sidebar configuration
st.sidebar.image(
    "https://img.icons8.com/clouds/200/000000/home.png",
    width=80
)
st.sidebar.title(" Navigation")
st.sidebar.markdown("---")

# Main content
st.markdown("""
    <div class="header">
        <h1>🏡 House Price Prediction</h1>
        <p class="subtitle">Predict house prices using advanced machine learning techniques</p>
    </div>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <h2>Welcome to the House Price Prediction App</h2>
            <p>An end-to-end machine learning application for accurate real estate price predictions</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Features section
st.markdown("## ✨ Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 EDA")
    st.markdown("Explore and visualize housing data with interactive charts and statistics.")

with col2:
    st.markdown("### ⚙️ Feature Engineering")
    st.markdown("Preprocess and transform your data for optimal model performance.")

with col3:
    st.markdown("### 🤖 Model Training")
    st.markdown("Train and compare multiple machine learning models.")

# How to use section
st.markdown("""
    ## 🚀 Getting Started
    1. **Start with EDA** - Explore the dataset and understand the features
    2. **Feature Engineering** - Preprocess and transform your data
    3. **Model Training** - Train and evaluate different models
    4. **Prediction** - Make predictions with the trained model
    5. **Summary** - View project insights and results
""")

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit | House Price Prediction App</p>
    </div>
""", unsafe_allow_html=True)

# Add custom CSS if style.css doesn't exist
if not Path("style.css").exists():
    with open("style.css", "w") as f:
        f.write("""
        /* Global Styles */
        body {
            color: #4a4a4a;
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Header Styles */
        .header {
            padding: 2rem 0;
            text-align: center;
        }
        
        .header h1 {
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: #7f8c8d;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        /* Hero Section */
        .hero {
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            color: white;
            padding: 3rem 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .hero-content h2 {
            color: white;
            margin-bottom: 1rem;
        }
        
        /* Cards */
        .st-eb {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            height: 100%;
            transition: transform 0.3s ease;
        }
        
        .st-eb:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 1rem;
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        
        /* Sidebar */
        .css-1d391kg, .css-1vq4p4l {
            background-color: #2c3e50 !important;
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 20px;
            border: none;
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            color: white;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        """)
