import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURATION ---
st.set_page_config(page_title="Ford Car Price Predictor", layout="centered")

# --- LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_artifacts():
    # Load the dictionary containing model, scaler, and encoders
    artifacts = joblib.load('car_price_model.pkl')
    return artifacts

try:
    artifacts = load_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    encoders = artifacts['encoders']
    feature_order = artifacts['features']
except FileNotFoundError:
    st.error("Error: 'car_price_model.pkl' not found. Please make sure the file is in the same folder as app.py")
    st.stop()

# --- APP UI ---
st.title("🚗 Ford Car Price Prediction")
st.write("Enter the car details below to get an estimated selling price.")

# create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # 1. Model Selection
    known_models = encoders['model'].classes_
    selected_model = st.selectbox("Car Model", known_models)

    # 2. Year
    year = st.number_input("Registration Year", min_value=1990, max_value=2024, value=2018)

    # 3. Transmission
    known_trans = encoders['transmission'].classes_
    transmission = st.selectbox("Transmission Type", known_trans)
    
    # 4. Mileage
    mileage = st.number_input("Mileage (miles)", min_value=0, value=20000)

with col2:
    # 5. Fuel Type
    known_fuel = encoders['fuelType'].classes_
    fuelType = st.selectbox("Fuel Type", known_fuel)
    
    # 6. Tax
    tax = st.number_input("Road Tax (£)", min_value=0, value=150)

    # 7. MPG
    mpg = st.number_input("MPG (Miles Per Gallon)", min_value=0.0, value=55.0)

    # 8. Engine Size
    engineSize = st.number_input("Engine Size (Litres)", min_value=0.0, value=1.5, step=0.1)

# --- PREDICTION LOGIC ---
if st.button("Predict Price"):
    try:
        # 1. Encode Categorical Inputs using the saved encoders
        model_enc = encoders['model'].transform([selected_model])[0]
        trans_enc = encoders['transmission'].transform([transmission])[0]
        fuel_enc = encoders['fuelType'].transform([fuelType])[0]

        # 2. Create the DataFrame matching the training data order
        # The columns must match the order expected by the scaler exactly
        input_data = pd.DataFrame([[
            model_enc, year, trans_enc, mileage, fuel_enc, tax, mpg, engineSize
        ]], columns=feature_order)

        # 3. Scale the data
        input_data_scaled = scaler.transform(input_data)
        
        # 4. Convert back to DataFrame to fix the "valid feature names" warning
        input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=feature_order)

        # 5. Predict
        prediction = model.predict(input_data_scaled_df)
        
        # 6. Display Result
        price = prediction[0]
        st.success(f"### Estimated Price: £{price:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- SIDEBAR INFO ---
st.sidebar.header("About")
st.sidebar.info("This app uses a Linear Regression model trained on Ford car data to estimate market value.")