import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Income Estimator", layout="centered")

st.title("🧮 Income & Repayment Capability Estimator")

st.markdown("Provide the required inputs to estimate monthly income:")

# --- Input fields ---
state = st.text_input("State Name", "Bihar")
district = st.text_input("District", "Patna")
pincode = st.number_input("Pincode", min_value=100000, max_value=999999, value=800001)

night_light = st.number_input("Night Light Intensity", value=10.0)
upi = st.number_input("UPI Transactions Per Capita", value=5.0)
mobile_pen = st.slider("Mobile Penetration (%)", 0.0, 100.0, 75.0)
literacy = st.slider("Literacy Rate (%)", 0.0, 100.0, 70.0)
microbiz = st.number_input("Microbusiness Count Per 1000 People", value=15.0)
distance = st.number_input("Distance to City (km)", value=12.0)
internet_score = st.slider("Internet Availability Score", 0.0, 1.0, 0.8)
ecommerce_freq = st.number_input("E-commerce Transaction Frequency", value=4.0)
mgnrega_days = st.number_input("MGNREGA Days Worked Last Year", value=45)
recharge_avg = st.number_input("Mobile Recharge Average (INR)", value=100.0)
healthcare_index = st.slider("Healthcare Access Index", 0.0, 1.0, 0.7)

# --- Submit button ---
submitted = st.button("Predict Income")

# --- Load model ---
model = joblib.load("models/income_model.pkl")

if submitted:
    input_df = pd.DataFrame([{
        "StateName": state,
        "District": district,
        "Pincode": pincode,
        "Night_Light_Intensity": night_light,
        "UPI_Transactions_Per_Capita": upi,
        "Mobile_Penetration_Percent": mobile_pen,
        "Literacy_Rate_Percent": literacy,
        "Microbusiness_Count_Per_1000": microbiz,
        "Distance_to_City_km": distance,
        "Internet_Availability_Score": internet_score,
        "Ecommerce_Transaction_Frequency": ecommerce_freq,
        "MGNREGA_Days_Worked_Last_Year": mgnrega_days,
        "Mobile_Recharge_Avg_INR": recharge_avg,
        "Healthcare_Access_Index": healthcare_index
    }])

    predicted_log_income = model.predict(input_df)[0]
    predicted_income = round(np.expm1(predicted_log_income), 2)

    st.success(f"🤑 **Predicted Monthly Income:** ₹{predicted_income}")
