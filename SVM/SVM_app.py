import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title='Smart Loan Approval System',
    layout='centered'
)

st.markdown("""
<style>
body { background-color: #f5f7fb; }
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
.hero {
    text-align: center;
    color: white;
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 30px;
}
.gradient-hero {
    background: linear-gradient(135deg, #4e54c8, #8f94fb);
}
.gradient-green {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
}
.gradient-red {
    background: linear-gradient(135deg, #cb2d3e, #ef473a);
    color: white;
}
.prediction-box {
    font-size: 22px;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
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
<div class="hero gradient-hero">
  <h1>🏦 Smart Loan Approval System</h1>
  <p>This system uses <b>Support Vector Machines (SVM)</b> to predict loan approval</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
    df.drop("Loan_ID", axis=1, inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    df = pd.get_dummies(df, drop_first=True)
    return df


df = load_data()

X = df.drop("Loan_Status_Y", axis=1)
y = df["Loan_Status_Y"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

svm_linear = SVC(kernel="linear", C=1)
svm_poly   = SVC(kernel="poly", degree=3, C=1)
svm_rbf    = SVC(kernel="rbf", C=1, gamma="scale")

svm_linear.fit(X_train_s, y_train)
svm_poly.fit(X_train_s, y_train)
svm_rbf.fit(X_train_s, y_train)

acc_linear = accuracy_score(y_test, svm_linear.predict(X_test_s))
acc_poly   = accuracy_score(y_test, svm_poly.predict(X_test_s))
acc_rbf    = accuracy_score(y_test, svm_rbf.predict(X_test_s))

st.subheader("📈 Model Accuracy Comparison")

st.write(f"🔹 **Linear SVM Accuracy:** {acc_linear * 100:.2f}%")
st.write(f"🔹 **Polynomial SVM Accuracy:** {acc_poly * 100:.2f}%")
st.write(f"🔹 **RBF SVM Accuracy:** {acc_rbf * 100:.2f}%")

best_model = max(
    [("Linear SVM", acc_linear), ("Polynomial SVM", acc_poly), ("RBF SVM", acc_rbf)],
    key=lambda x: x[1]
)

st.success(f"🏆 Best Performing Model: **{best_model[0]}**")

st.divider()

st.sidebar.header("Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=500)
credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment_status = st.sidebar.selectbox(
    "Employment Status", ["Employed", "Self-Employed", "Unemployed"]
)
property_area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

st.sidebar.header("⚙️ Model Selection")
kernel_options = st.sidebar.radio(
    "Select Kernel", ["Linear", "Polynomial", "RBF"]
)

user_dict = {
    "ApplicantIncome": income,
    "CoapplicantIncome": 0,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": 360,
    "Credit_History": 1 if credit_history == "Yes" else 0,
    "Self_Employed_Yes": 1 if employment_status == "Self-Employed" else 0,
    "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
    "Property_Area_Urban": 1 if property_area == "Urban" else 0
}

user_df = pd.DataFrame([user_dict])

for col in feature_names:
    if col not in user_df.columns:
        user_df[col] = 0

user_df = user_df[feature_names]

if kernel_options == "Linear":
    model = SVC(kernel="linear", C=1)
elif kernel_options == "Polynomial":
    model = SVC(kernel="poly", degree=3, C=1)
else:
    model = SVC(kernel="rbf", C=1, gamma="scale")

model.fit(X_train_s, y_train)

if st.button("🔍 Check Loan Eligibility"):
    scaled_input = scaler.transform(user_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.markdown("<div class='card gradient-green prediction-box'>✅ Loan Approved</div>", unsafe_allow_html=True)
        st.info("Based on credit history and income pattern, the applicant is likely to repay the loan.")
    else:
        st.markdown("<div class='card gradient-red prediction-box'>❌ Loan Rejected</div>", unsafe_allow_html=True)
        st.warning("Based on credit history and income pattern, the applicant is unlikely to repay the loan.")

    st.markdown(f"""
    <div class="card">
    <b>Kernel Used:</b> {kernel_options}
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  Smart Loan Approval System using Support Vector Machines
</div>
""", unsafe_allow_html=True)
