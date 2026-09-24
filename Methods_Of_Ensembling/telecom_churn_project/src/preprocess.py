import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


def preprocess_data(df_filepath):

    df = pd.read_csv(df_filepath)

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    ordinal_columns = ['Contract']
    nominal_columns = [
        'gender',
        'SeniorCitizen',
        'Partner',
        'Dependents',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'PaperlessBilling',
        'PaymentMethod',
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    contract_categories = [['Month-to-month', 'One year', 'Two year']]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                'numeric_transform',
                make_pipeline(
                    SimpleImputer(strategy='median')
                ),
                numeric_columns,
            ),
            (
                'nominal_transform',
                make_pipeline(
                    SimpleImputer(strategy='most_frequent'),
                    OneHotEncoder(drop='first', sparse_output=False),
                ),
                nominal_columns,
            ),
            (
                'ordinal_transform',
                make_pipeline(
                    SimpleImputer(strategy='most_frequent'),
                    OrdinalEncoder(categories=contract_categories),
                ),
                ordinal_columns,
            ),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessing