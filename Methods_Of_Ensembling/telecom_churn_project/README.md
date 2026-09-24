# 📊 Telecom Customer Churn Predictor & Explainability Engine

An end-to-end Machine Learning web application that predicts customer churn risk for a telecommunications company and breaks down model decisions using SHAP (SHapley Additive exPlanations).

🚀 **[Live Interactive Web Application](https://telecom-customer-i6hgibrsqs4ffpid9whwnq.streamlit.app/#telecom-customer-predictor)**

---

## 📌 Features

- **Real-Time Predictions:** Interactive sidebar inputs for all 19 customer features across Demographics, Account/Billing, and Telecom Services.
- **Explainable AI (XAI):** Integrated SHAP waterfall visualizations showing exact positive and negative feature contributions per customer profile.
- **Production-Ready Pipeline:** Serialized Scikit-Learn Random Forest model bundled into a clean, modular structure.

---

## 📂 Project Structure

```text
telecom_churn_project/
├── notebooks/                 # Iterative data science workflow
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Feature cleaning & transformation
│   ├── 03_modeling.ipynb      # Model training & hyperparameter tuning
│   └── 04_evalutation.ipynb   # SHAP & model evaluation metrics
├── models/
│   └── random_forest.joblib   # Serialized ML pipeline
├── src/
│   ├── preprocess.py          # Feature engineering & custom pipeline logic
│   └── evaluate.py            # SHAP explanation & plot rendering routines
├── app.py                     # Streamlit web application dashboard
└── requirements.txt           # Dependencies for cloud deployment
```

---

## 🛠️ Tech Stack

- **Python 3.10**
- **Streamlit** (UI & Cloud Hosting)
- **Scikit-Learn** (Random Forest Classifier & Pipelines)
- **SHAP** (Model Explainability)
- **Pandas & NumPy** (Data Manipulation)

---

## 🚀 Local Run Instructions

To run this project locally:

1. Open your terminal and navigate to this folder:
   ```powershell
   cd "Methods_Of_Ensembling/telecom_churn_project"
   ```

2. Install the required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Launch the Streamlit application:
   ```powershell
   streamlit run app.py
   ```