from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(title="Credit Risk API")

# Load model and features
model = joblib.load(r"C:\Users\danie\OneDrive\Desktop\Projects_Data\Project_6_Credit_Risk\models\catboost_final.pkl")
# CRITICAL: This list MUST be the full feature set used to train the CatBoost model, 
# and in the exact order the DataFrame columns were when CatBoost was fit.
FULL_TRAINING_FEATURES = [
    'age', 'late_30_59', 'open_credit', 'late_90', 'real_estate', 'late_60_89', 
    'dependents', 'rev_util_log', 'debt_ratio_log', 'monthly_inc_log', 
    'income_to_debt', 'credit_utilization', 'income_per_dependent'
]
selected_features = FULL_TRAINING_FEATURES # We will now use this list for ordering
class LoanApplication(BaseModel):
    rev_util: float
    age: float
    late_30_59: float
    debt_ratio: float
    monthly_inc: float
    open_credit: float
    late_90: float
    real_estate: float
    late_60_89: float
    dependents: float

@app.post("/predict")
def predict(data: LoanApplication):
    # Convert Pydantic model to DataFrame
    input_df = pd.DataFrame([data.model_dump()])  
    
    # 1. Apply all required transformations/feature engineering
    # CRITICAL: These features MUST be created before selecting the final set.
    # Note: These are the three features you dropped later, which is why we must log them now.
    input_df['rev_util_log'] = np.log1p(input_df['rev_util'])
    input_df['debt_ratio_log'] = np.log1p(input_df['debt_ratio'])
    input_df['monthly_inc_log'] = np.log1p(input_df['monthly_inc'])
    
    # These are the three new engineered features that the model uses
    input_df['income_to_debt'] = input_df['monthly_inc'] / (input_df['debt_ratio'] + 1)
    input_df['credit_utilization'] = input_df['rev_util'] / (input_df['open_credit'] + 1)
    input_df['income_per_dependent'] = input_df['monthly_inc'] / (input_df['dependents'] + 1)

    # 2. CRITICAL FIX: Filter and explicitly re-order the columns to match CatBoost's exact requirement
    try:
        # This step selects the columns used by the model (base features + engineered features) 
        # and enforces the exact order saved in selected_features.pkl.
        df_model = input_df[selected_features] 

    except KeyError as e:
        # This will catch if one of the log features or new engineered features is missing
        return {"error": f"Internal Feature Error: Feature {e} is missing after engineering."}

    # ===============================================
    #       <<< INSERT DEBUGGING CODE HERE >>>
    # ===============================================
    print("\nDEBUGGING INFO:")
    print("Expected Feature Order (selected_features):")
    print(selected_features)
    print("\nActual DataFrame Column Order (df_model.columns):")
    print(df_model.columns.tolist())
    # ===============================================
    
    
    # Predict
    prediction = model.predict(df_model)[0]
    probability = model.predict_proba(df_model)[0][1]
    
    # Assign risk level
    if probability > 0.6:
        risk = "HIGH"
    elif probability > 0.3:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    
    return {
        "default_prediction": int(prediction),
        "default_probability": float(probability),
        "risk_level": risk,
        "recommendation": "REJECT" if prediction == 1 else "APPROVE"
    }

@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
