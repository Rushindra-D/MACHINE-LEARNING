import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# ---------------- INLINE STYLING ----------------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    background: #ffffff;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
.hero {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    text-align: center;
}
.metric-box {
    background: #f4f6f9;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
.prediction-box {
    background: #e8f5e9;
    color: #2e7d32;
    padding: 15px;
    border-radius: 10px;
    font-size: 18px;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\dobil\\OneDrive\\Desktop\\Logistic_Regression\\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

df = load_data()

st.markdown("""
<div class="card hero">
    <h1>📊 Customer Churn Prediction</h1>
    <p>Logistic Regression Classification Model</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📁 Dataset Preview")
st.dataframe(df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].head(10),
             use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

X = df[["tenure", "MonthlyCharges", "TotalCharges"]].values
y = df["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧮 Confusion Matrix")

fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

col1, col2 = st.columns([1, 2])
with col1:
    st.pyplot(fig)

with col2:
    st.markdown(f"""
    **True Negatives:** {cm[0][0]}  
    **False Positives:** {cm[0][1]}  
    **False Negatives:** {cm[1][0]}  
    **True Positives:** {cm[1][1]}  

    📌 *False Negatives are costly because churn customers are missed.*
    """)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎯 Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.2f}")
c2.metric("Precision", f"{report['1']['precision']:.2f}")
c3.metric("Recall", f"{report['1']['recall']:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧪 Predict for a New Customer")

tenure = st.number_input("Tenure (Months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", value=70.0)
total = st.number_input("Total Charges", value=800.0)

if st.button("Predict Churn"):
    input_scaled = scaler.transform([[tenure, monthly, total]])
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    st.markdown(
        f"<div class='prediction-box'>📈 Churn Probability: <b>{prob:.2f}</b></div>",
        unsafe_allow_html=True
    )

    if pred == 1:
        st.error("🚨 Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
Logistic Regression | Customer Churn Prediction App
</div>
""", unsafe_allow_html=True)
