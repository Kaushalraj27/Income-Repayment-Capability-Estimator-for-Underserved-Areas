import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# Load data
data = pd.read_csv("E:/Data Science/lenden hacka/data/pincode_enriched.csv")
print("Available columns:", list(data.columns))
print("shape=", data.shape)

# Drop rows with missing target
data = data.dropna(subset=["Predicted_Monthly_Income_INR"])

# Define features and target
target_col = "Predicted_Monthly_Income_INR"
categorical_features = ["StateName", "District", "Pincode"]
numerical_features = [
    "Night_Light_Intensity",
    "UPI_Transactions_Per_Capita",
    "Mobile_Penetration_Percent",
    "Literacy_Rate_Percent",
    "Microbusiness_Count_Per_1000",
    "Distance_to_City_km",
    "Internet_Availability_Score",
    "Ecommerce_Transaction_Frequency",
    "MGNREGA_Days_Worked_Last_Year",
    "Mobile_Recharge_Avg_INR",
    "Healthcare_Access_Index"
]

X = data[categorical_features + numerical_features]
y = data[target_col]

# Log transform the target to stabilize variance
y_log = np.log1p(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", TargetEncoder(), categorical_features)
])

# Pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred_log = model_pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Reverse log1p
y_true = np.expm1(y_test)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# Save model
joblib.dump(model_pipeline, "trained_income_model.pkl")
print("Model saved to trained_income_model.pkl")
