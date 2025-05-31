import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("trained_income_model.pkl")
print("Model loaded.")

# Example input (can be replaced by CSV loading or dynamic input)
new_data = pd.DataFrame([{
    "StateName": "Uttar Pradesh",
    "District": "Lucknow",
    "Pincode": 226001,
    "Night_Light_Intensity": 35.2,
    "UPI_Transactions_Per_Capita": 23.5,
    "Mobile_Penetration_Percent": 87.6,
    "Literacy_Rate_Percent": 82.1,
    "Microbusiness_Count_Per_1000": 5.4,
    "Distance_to_City_km": 2.5,
    "Internet_Availability_Score": 0.85,
    "Ecommerce_Transaction_Frequency": 1.7,
    "MGNREGA_Days_Worked_Last_Year": 12,
    "Mobile_Recharge_Avg_INR": 155.0,
    "Healthcare_Access_Index": 0.65
}])

# Predict (model expects dataframe)
log_income_pred = model.predict(new_data)
income_pred = np.expm1(log_income_pred)  # Reverse log1p

print(f"Predicted Monthly Income (INR): {income_pred[0]:,.2f}")
