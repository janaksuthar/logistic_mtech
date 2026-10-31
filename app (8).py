import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('logistic.sav')

st.title('🚚 Delivery Delay Predictor')
st.write('Fill in the details below to predict if a delivery will be delayed.')

# Feature Inputs
col1, col2 = st.columns(2)

with col1:
    delivery_distance = st.number_input('Delivery Distance (km)', min_value=0.0, value=25.0)
    traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
    weather_condition = st.slider('Weather Condition (1-3)', 1, 3, 2)
    delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
    driver_experience = st.number_input('Driver Experience (years)', min_value=0, value=5)
    num_stops = st.number_input('Number of Stops', min_value=0, value=3)

with col2:
    vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, value=3)
    road_condition = st.slider('Road Condition Score (1-5)', 1, 5, 3)
    package_weight = st.number_input('Package Weight (kg)', min_value=0.0, value=10.0)
    fuel_efficiency = st.number_input('Fuel Efficiency (km/l)', min_value=0.0, value=12.0)
    warehouse_time = st.number_input('Warehouse Processing Time (mins)', min_value=0, value=45)

# Prediction logic
if st.button('Predict Delay'):
    features = pd.DataFrame([[
        delivery_distance, traffic_congestion, weather_condition, delivery_slot,
        driver_experience, num_stops, vehicle_age, road_condition,
        package_weight, fuel_efficiency, warehouse_time
    ]], columns=[
        'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot',
        'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 'Road_Condition_Score',
        'Package_Weight', 'Fuel_Efficiency', 'Warehouse_Processing_Time'
    ])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    if prediction == 1:
        st.error(f'⚠️ Delay Predicted with a probability of {probability:.2%}')
    else:
        st.success(f'✅ On-time delivery predicted with a probability of {(1 - probability):.2%}')