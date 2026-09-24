from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

# 1. Resolve Dynamic Directory Paths & append to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# 2. Import Custom Modules
import src.preprocess
from src.evaluate import plot_shap

MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"

# 3. Page Configuration
st.set_page_config(
    page_title="Telecommunication Customer Prediction", page_icon="🔮", layout="wide"
)

st.title("📊 Telecom Customer Predictor")
st.write(
    "Adjust customer features on the sidebar to get real-time customer predictions and SHAP decision breakdowns."
)


# 4. Load Model Pipeline
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    pipeline = load_model()
except Exception as e:
    st.error(
        f"Could not load model file from '{MODEL_PATH}'. Error details: {e}"
    )
    st.stop()

# 5. Sidebar Input Form (Organized in Expanders)
st.sidebar.header("📋 Customer Profile")

# --- Demographics ---
with st.sidebar.expander("👤 Demographics", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

# --- Account & Billing (Numeric & Contract Controls) ---
with st.sidebar.expander("💳 Account & Billing", expanded=True):
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    contract = st.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)", min_value=0, max_value=120.0, value=65.0, step=1.0
    )
    
    # Auto-calculates default baseline, but allows full continuous numeric input/editing
    default_total = float(tenure * monthly_charges)
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10000.0, value=default_total, step=10.0
    )

# --- Telecom Services ---
with st.sidebar.expander("🛠 Services & Add-ons", expanded=False):
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Construct input row matching all 19 raw features from training
input_df = pd.DataFrame(
    [
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
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
            "TotalCharges": total_charges,
        }
    ]
)

# 6. Prediction & Output
if st.button("🚀 Analyze if customer leaves", type="primary"):
    churn_proba = pipeline.predict_proba(input_df)[0][1]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Prediction Output")
        st.metric(label="Churn Probability", value=f"{churn_proba:.1%}")

        if churn_proba >= 0.50:
            st.error("⚠️ High Risk: Customer is likely to leave the company!")
        else:
            st.success("✅ Low Risk: Customer is likely to stay.")

    with col2:
        st.subheader("Model Decision Breakdown (SHAP)")
        with st.spinner("Generating SHAP explanation..."):
            fig = plot_shap(pipeline, input_df)
            st.pyplot(fig)