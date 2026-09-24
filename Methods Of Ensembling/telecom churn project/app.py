import joblib
import pandas as pd
import streamlit as st
from src.evaluate import plot_shap

# 1. Page Configuration
st.set_page_config(
    page_title="Telco Churn Predictor", page_icon="🔮", layout="wide"
)

st.title("📊 Telco Customer Churn Predictor & Explainability Engine")
st.write(
    "Adjust customer features on the sidebar to get real-time churn risk predictions and SHAP decision breakdowns."
)


# 2. Load Model Pipeline
@st.cache_resource
def load_model():
    # Update this filename if your saved model in models/ uses a different name
    return joblib.load("models/random_forest.joblib")


try:
    pipeline = load_model()
except Exception as e:
    st.error(
        f"Could not load model file from models/ directory. Error details: {e}"
    )
    st.stop()

# 3. Sidebar Input Form
st.sidebar.header("📋 Customer Profile")

tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
contract = st.sidebar.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)
internet_service = st.sidebar.selectbox(
    "Internet Service", ["Fiber optic", "DSL", "No"]
)
monthly_charges = st.sidebar.number_input(
    "Monthly Charges ($)", 18.0, 120.0, 65.0
)
total_charges = tenure * monthly_charges

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
tech_support = st.sidebar.selectbox(
    "Tech Support", ["No", "Yes", "No internet service"]
)
online_security = st.sidebar.selectbox(
    "Online Security", ["No", "Yes", "No internet service"]
)

# Demographics
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

# Construct input row matching exact raw column names from training
input_df = pd.DataFrame(
    [
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": tech_support,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
    ]
)

# 4. Prediction & Output
if st.button("🚀 Analyze Churn Risk", type="primary"):
    churn_proba = pipeline.predict_proba(input_df)[0][1]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Prediction Output")
        st.metric(label="Churn Probability", value=f"{churn_proba:.1%}")

        if churn_proba >= 0.50:
            st.error("⚠️ High Risk: Customer is likely to churn!")
        else:
            st.success("✅ Low Risk: Customer is likely to stay.")

    with col2:
        st.subheader("Model Decision Breakdown (SHAP)")
        with st.spinner("Generating SHAP explanation..."):
            # Calls plot_shap from src/evaluate.py
            fig = plot_shap(pipeline, input_df)
            st.pyplot(fig)