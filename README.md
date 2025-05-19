<div align="center">
  <h1>🏠 House Hunting</h1>
  <p>
    Your intelligent home-buying companion - Predict house prices and find your dream home with AI-powered insights
    <br />
    <a href="" target="_blank">View Demo</a>
    ·
    <a href="#features">Features</a>
    ·
    <a href="#getting-started">Getting Started</a>
    ·
    <a href="#deployment">Deployment</a>
  </p>
  
  [![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://house-prediction-wctwgkvhjgqgnkktz9tyuc.streamlit.app/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
</div>

## ✨ Features

- **🏡 Smart Price Prediction**: Get instant, accurate house price estimates using advanced ML models
- **📊 Interactive Dashboard**: Beautiful and intuitive interface for exploring properties
- **🔍 Market Insights**: Comprehensive data visualization and neighborhood analysis
- **🤖 AI-Powered**: Multiple regression models (XGBoost, Random Forest, and more) for reliable predictions
- **📱 Responsive Design**: Seamless experience across all devices
- **🔧 Feature Analysis**: Understand what factors influence property prices the most

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- (Optional) Docker

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ruchi2117/house-hunting.git
   cd house-hunting
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```
   The app will be available at `http://localhost:8501`

## 🐳 Docker Deployment

You can also run the application using Docker:

```bash
# Build the Docker image
docker build -t house-price-prediction .

# Run the container
docker run -p 8501:8501 house-price-prediction
```

## 📊 Project Structure

```
house-price-prediction/
├── data/                    # Dataset directory
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed data files
├── pages/                   # Streamlit pages
│   ├── 0_Summary.py         # Project overview
│   ├── 1_EDA.py            # Exploratory Data Analysis
│   ├── 2_Feature_Engineering.py  # Feature engineering
│   ├── 3_Model_Training.py  # Model training and evaluation
│   └── 4_Prediction.py      # Price prediction interface
├── .gitignore
├── app.py                   # Main application
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools
- [XGBoost](https://xgboost.ai/) for gradient boosting
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualizations

---

<div align="center">
  <p>Crafted with ❤️ by Ruchi Shaktawat 🚀</p>
  <p>Thank you for checking out House Hunting! If you have any feedback or suggestions, feel free to reach out. May you find your dream home! 🏡</p>
</div>
