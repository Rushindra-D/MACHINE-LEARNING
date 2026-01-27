import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
st.set_page_config(
    page_title="Customer Risk Prediction System (KNN)",
    layout="centered"
)
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
.header {
    text-align: center;
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    margin-bottom: 30px;
}
.header h1 {
    font-size: 36px;
}
.header p {
    font-size: 18px;
}
.result-high {
    background: linear-gradient(135deg, #cb2d3e, #ef473a);
    color: white;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}
.result-low {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}
.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="header">
    <h1>Customer Risk Prediction System (KNN)</h1>
    <p>This system predicts customer risk by comparing them with similar customers.</p>
</div>
""", unsafe_allow_html=True)
@st.cache_data
def load_data():
    if not os.path.exists("credit_risk_dataset.csv"):
        st.error("Dataset file not found. Please upload credit_risk_dataset.csv")
        st.stop()
    return pd.read_csv("credit_risk_dataset.csv")
df = load_data()
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)
df_encoded = pd.get_dummies(
    df,
    columns=[
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file"
    ],
    drop_first=True
)
X = df_encoded.drop("loan_status", axis=1)
y = df_encoded["loan_status"]
feature_names = X.columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

st.sidebar.header("👤 Customer Input")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Annual Income", min_value=1000, value=50000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=500, value=15000)
credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])

k_value = st.sidebar.slider("K Value (No. of Neighbors)", 1, 15, 5)
user_data = {
    "person_age": age,
    "person_income": income,
    "person_emp_length": 5,
    "loan_amnt": loan_amount,
    "loan_int_rate": 12.0,
    "loan_percent_income": loan_amount / income,
    "cb_person_cred_hist_length": 6,
    "person_home_ownership": "RENT",
    "loan_intent": "EDUCATION",
    "loan_grade": "B",
    "cb_person_default_on_file": "N" if credit_history == "Yes" else "Y"
}

user_df = pd.DataFrame([user_data])
user_encoded = pd.get_dummies(
    user_df,
    columns=[
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file"
    ],
    drop_first=True
)

user_encoded = user_encoded.reindex(columns=feature_names, fill_value=0)
user_scaled = scaler.transform(user_encoded)

knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train, y_train)

if st.button("🔍 Predict Customer Risk"):

    prediction = knn.predict(user_scaled)[0]
    distances, indices = knn.kneighbors(user_scaled)

    neighbor_labels = y_train.iloc[indices[0]]
    high_risk_count = sum(neighbor_labels)
    low_risk_count = len(neighbor_labels) - high_risk_count

    if prediction == 1:
        st.markdown("<div class='result-high'>🔴 High Risk Customer</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-low'>🟢 Low Risk Customer</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📍 Nearest Neighbors Explanation")

    st.write(f"**K (Neighbors Used):** {k_value}")
    st.write(f"**High Risk Neighbors:** {high_risk_count}")
    st.write(f"**Low Risk Neighbors:** {low_risk_count}")

    neighbor_table = df.iloc[indices[0]][
        ["person_age", "person_income", "loan_amnt", "loan_status"]
    ]

    st.write("📋 Similar Customers:")
    st.dataframe(neighbor_table)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💡 Business Insight")
    st.info(
        "This decision is based on similarity with nearby customers in feature space. "
        "The system evaluates how similar customers behaved and assigns risk based on "
        "majority behavior among the nearest neighbors."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
Customer Risk Prediction System using K-Nearest Neighbors
</div>
""", unsafe_allow_html=True)
