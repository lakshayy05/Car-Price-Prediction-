#  Car Price Prediction 🚗

A full-stack Machine Learning application that predicts the resale value of Ford cars based on their features. This project includes a comprehensive **Jupyter Notebook** for data analysis and model training, and a user-friendly **Streamlit Web App** for real-time predictions.

## 🚀 Project Overview

Buying or selling a used car can be tricky without knowing its fair market value. This tool solves that problem by using historical data to estimate prices based on key factors like model, mileage, year, and fuel type.

## 📂 Project Structure

The repository is organized as follows:

```text
Car-Price-Prediction/
│
├── app.py                       # 🖥️ Frontend: Streamlit Web Application
├── car_price_prediction.ipynb   # 📓 Backend: EDA, Preprocessing & Model Training
├── car_price_model.pkl          # 📦 Artifacts: Saved Model, Scaler, and Encoders
├── requirements.txt             # ⚙️ Dependencies: List of required Python libraries
└── README.md                    # 📄 Documentation

📊 Workflow & Features
1. Data Analysis (Jupyter Notebook)
EDA: Performed in-depth Exploratory Data Analysis using histograms, heatmaps, and boxplots to understand price distributions and correlations.
Preprocessing:
Label Encoding: Converted categorical features (model, transmission, fuelType) into numerical format.
Feature Scaling: Applied StandardScaler to normalize numerical columns (year, mileage, tax, mpg, engineSize) for better model performance.
Model Training: Trained a Linear Regression model to predict continuous price values.
Evaluation: Achieved a strong R² score, confirming the model's accuracy.

2. Web Application (Streamlit)
User Interface: A clean, interactive form where users can select car details from dropdowns.
Real-Time Prediction: The app loads the trained model (.pkl) to generate instant price estimates.
Robust Handling: Automatically handles data scaling and encoding behind the scenes, so the user only sees simple text options.

🛠️ Tech Stack
Language: Python 3.13.3
Machine Learning: Scikit-Learn (Linear Regression)
Data Manipulation: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Web Framework: Streamlit
