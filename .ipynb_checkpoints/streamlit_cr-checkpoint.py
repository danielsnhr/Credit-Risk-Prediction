import streamlit as st
import requests
import json # Added for robust error handling

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("🏦 Credit Risk Loan Default Prediction")
st.markdown("---")

# Use two main columns for input layout
col1, col2 = st.columns(2)

# --- COLUMN 1: APPLICANT & REAL ESTATE ---
with col1:
    st.subheader("📊 Applicant Information")
    age = st.slider("Age", 18, 100, 35)
    
    # Use float for consistency with Pydantic model
    monthly_inc = st.number_input("Monthly Income ($)", 0.0, 250000.0, 5000.0, step=100.0)
    
    dependents = st.slider("Number of Dependents", 0, 8, 0)
    
    # ADDED: Missing feature required by FastAPI model
    real_estate = st.slider("Number of Real Estate Loans/Lines (Mortgages)", 0, 5, 1) 

# --- COLUMN 2: CREDIT PROFILE ---
with col2:
    st.subheader("💳 Credit Profile")
    
    # Input as percentage, convert to decimal for API payload
    rev_util_input = st.slider("Revolving Utilization (%)", 0.0, 100.0, 30.0) 
    rev_util = rev_util_input / 100.0
    
    debt_ratio_input = st.slider("Debt Ratio (%)", 0.0, 100.0, 30.0)
    debt_ratio = debt_ratio_input / 100.0
    
    open_credit = st.slider("Open Credit Lines", 0, 50, 8)
    
    # ADDED: Missing features required by FastAPI model
    late_30_59 = st.number_input("30-59 Days Past Due (Excl. 90+)", 0, 20, 0) 
    late_60_89 = st.number_input("60-89 Days Past Due (Excl. 90+)", 0, 20, 0)
    late_90 = st.number_input("90+ Days Late Payments", 0, 20, 0) # Already present

if st.button("🎯 Predict Default Risk", type="primary"):
    
    # 1. Prepare data in the EXACT ORDER and with all 10 fields 
    # as required by the FastAPI LoanApplication class
    data = {
        "rev_util": rev_util,          # 1
        "age": float(age),             # 2
        "late_30_59": float(late_30_59), # 3
        "debt_ratio": debt_ratio,      # 4
        "monthly_inc": float(monthly_inc), # 5
        "open_credit": float(open_credit), # 6
        "late_90": float(late_90),       # 7
        "real_estate": float(real_estate), # 8
        "late_60_89": float(late_60_89), # 9
        "dependents": float(dependents)  # 10
    }
    
    try:
        # 2. Call API
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        
        st.markdown("---")
        st.header("📈 Prediction Results")

        # 3. CRITICAL FIX: Check status code and handle errors gracefully
        if response.status_code == 200:
            result = response.json()
            
            # Now we know result is a success JSON, so we display metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                # Assign color based on risk level
                risk_level = result['risk_level']
                if risk_level == "HIGH":
                    risk_color = "🔴"
                elif risk_level == "MEDIUM":
                    risk_color = "🟡"
                else:
                    risk_color = "🟢"
                    
                st.metric("Risk Level", f"{risk_color} {risk_level}")
                
            with col_b:
                st.metric("Default Probability", f"{result['default_probability']:.1%}")
            
            with col_c:
                rec_color = "🟢" if result['recommendation'] == "APPROVE" else "🔴"
                st.metric("Recommendation", f"{rec_color} {result['recommendation']}")

        else:
            # Handle API errors (422, 500, etc.)
            st.error(f"API Request Failed with status code: {response.status_code}")
            try:
                # Try to parse the error message from the API's JSON body
                error_data = response.json()
                # Check for the common FastAPI validation error structure
                if 'detail' in error_data:
                    st.warning(f"Validation Error: {error_data['detail'][0]['msg']}")
                # Check for the custom 500 error structure from your API
                elif 'error' in error_data:
                    st.warning(f"FastAPI Internal Error: {error_data['error']}")
                else:
                    st.json(error_data)
            except json.JSONDecodeError:
                # If the API returned plain text error (e.g., "Internal Server Error")
                st.warning(f"Server response was not JSON: {response.text}")


    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the FastAPI service. Ensure it is running at http://127.0.0.1:8000.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")