import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("🏦 Credit Risk Loan Default Prediction")
st.markdown("---")

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "catboost_final.pkl")
    return joblib.load(model_path)

model = load_model()

# -------------------------------
# Define features in the exact order
# -------------------------------
FULL_TRAINING_FEATURES = [
    'age', 'late_30_59', 'open_credit', 'late_90', 'real_estate', 'late_60_89',
    'dependents', 'rev_util_log', 'debt_ratio_log', 'monthly_inc_log',
    'income_to_debt', 'credit_utilization', 'income_per_dependent'
]

# -------------------------------
# Layout for inputs
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Applicant Information")
    age = st.slider("Age", 18, 100, 35)
    monthly_inc = st.number_input("Monthly Income ($)", 0.0, 250000.0, 5000.0, step=100.0)
    dependents = st.slider("Number of Dependents", 0, 8, 0)
    real_estate = st.slider("Number of Real Estate Loans/Lines (Mortgages)", 0, 5, 1) 

with col2:
    st.subheader("💳 Credit Profile")
    rev_util_input = st.slider("Revolving Utilization (%)", 0.0, 100.0, 30.0) 
    rev_util = rev_util_input / 100.0
    debt_ratio_input = st.slider("Debt Ratio (%)", 0.0, 100.0, 30.0)
    debt_ratio = debt_ratio_input / 100.0
    open_credit = st.slider("Open Credit Lines", 0, 50, 8)
    late_30_59 = st.number_input("30-59 Days Past Due (Excl. 90+)", 0, 20, 0) 
    late_60_89 = st.number_input("60-89 Days Past Due (Excl. 90+)", 0, 20, 0)
    late_90 = st.number_input("90+ Days Late Payments", 0, 20, 0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🎯 Predict Default Risk", type="primary"):

    # 1. Build input dataframe
    input_df = pd.DataFrame([{
        "rev_util": rev_util,
        "age": float(age),
        "late_30_59": float(late_30_59),
        "debt_ratio": debt_ratio,
        "monthly_inc": float(monthly_inc),
        "open_credit": float(open_credit),
        "late_90": float(late_90),
        "real_estate": float(real_estate),
        "late_60_89": float(late_60_89),
        "dependents": float(dependents)
    }])

    # 2. Feature engineering
    input_df['rev_util_log'] = np.log1p(input_df['rev_util'])
    input_df['debt_ratio_log'] = np.log1p(input_df['debt_ratio'])
    input_df['monthly_inc_log'] = np.log1p(input_df['monthly_inc'])
    input_df['income_to_debt'] = input_df['monthly_inc'] / (input_df['debt_ratio'] + 1)
    input_df['credit_utilization'] = input_df['rev_util'] / (input_df['open_credit'] + 1)
    input_df['income_per_dependent'] = input_df['monthly_inc'] / (input_df['dependents'] + 1)

    # 3. Reorder features
    df_model = input_df[FULL_TRAINING_FEATURES]

    # 4. Predict
    prediction = model.predict(df_model)[0]
    probability = model.predict_proba(df_model)[0][1]

    # 5. Risk assessment
    if probability > 0.6:
        risk = "HIGH"
        risk_icon = "🔴"
    elif probability > 0.3:
        risk = "MEDIUM"
        risk_icon = "🟡"
    else:
        risk = "LOW"
        risk_icon = "🟢"

    recommendation = "REJECT" if prediction == 1 else "APPROVE"
    rec_icon = "🟢" if recommendation == "APPROVE" else "🔴"

    # -------------------------------
    # Display results
    # -------------------------------
    st.markdown("---")
    st.header("📈 Prediction Results")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Risk Level", f"{risk_icon} {risk}")
    with col_b:
        st.metric("Default Probability", f"{probability:.1%}")
    with col_c:
        st.metric("Recommendation", f"{rec_icon} {recommendation}")
