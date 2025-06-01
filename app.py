import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessor
model = joblib.load("src/best_model.pkl")
preprocessor = joblib.load("src/preprocessor.pkl")

# Streamlit UI
st.title("🔍 Customer Churn Prediction App")
st.write("Enter customer details to predict if they'll churn.")

# User input fields
customer_id = st.text_input("Customer ID", "0000-A")
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)

# Build input dictionary
input_dict = {
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict Churn"):
    try:
        transformed_input = preprocessor.transform(input_df)
        prediction = model.predict(transformed_input)[0]
        probability = model.predict_proba(transformed_input)[0][1]

        if prediction == 1:
            st.error(f"⚠️ This customer is likely to churn. (Probability: {probability:.2f})")
        else:
            st.success(f"✅ This customer is likely to stay. (Probability: {1 - probability:.2f})")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
