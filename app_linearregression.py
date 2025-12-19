import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Linear Regression App", layout="centered")

def load_css(path):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

st.markdown("""
<div class="card hero gradient-hero">
  <div class="hero-title">📈 Linear Regression Model</div>
  <div class="hero-subtitle">
    Predict <b>Tip</b> from <b>Total Bill</b> using Machine Learning
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card gradient-blue">', unsafe_allow_html=True)
st.markdown("<h3>📊 Dataset Preview</h3>", unsafe_allow_html=True)
st.dataframe(df[["total_bill", "tip"]].head(10), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

X = df[["total_bill"]].to_numpy()
y = df["tip"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = r2_score(y_test, y_pred)

st.markdown('<div class="card gradient-purple">', unsafe_allow_html=True)
st.markdown("<h3>📉 Regression Visualization</h3>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(df["total_bill"], df["tip"], alpha=0.45)

x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_grid = model.predict(scaler.transform(x_grid))
ax.plot(x_grid, y_grid, linewidth=2.5)

ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
ax.grid(alpha=0.15)

st.pyplot(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card gradient-orange">', unsafe_allow_html=True)
st.markdown("<h3>🎯 Model Performance</h3>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("MSE", f"{mse:.2f}")
c3.metric("RMSE", f"{rmse:.2f}")
c4.metric("R²", f"{r2:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card gradient-green">', unsafe_allow_html=True)
st.markdown("<h3>🧮 Try Prediction</h3>", unsafe_allow_html=True)

bill = st.number_input("Enter Total Bill", value=30.0)

if st.button("Predict Tip"):
    tip = model.predict(scaler.transform([[bill]]))[0]
    st.markdown(
        f"<div class='prediction-box'>💰 Predicted Tip: <b>${tip:.2f}</b></div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
   Linear Regression model 
</div>
""", unsafe_allow_html=True)
