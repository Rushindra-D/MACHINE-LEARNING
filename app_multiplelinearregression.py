import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Multiple Linear Regression App", layout="centered")

def load_css(path):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles1.css")

@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

st.markdown("""
<div class="card hero gradient-hero">
  <div class="hero-title">📈 Multiple Linear Regression</div>
  <div class="hero-subtitle">
    Predict <b>Tip</b> using <b>Total Bill</b> & <b>Size</b>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card gradient-blue">', unsafe_allow_html=True)
st.markdown("<h3>📊 Dataset Preview</h3>", unsafe_allow_html=True)
st.dataframe(df[["total_bill", "size", "tip"]].head(10), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

X = df[["total_bill", "size"]].to_numpy()
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

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h1>Model Visualization</h1>', unsafe_allow_html=True)

plt.figure(figsize=(7, 4))

plt.title("Total Bill vs Tip Amount with Multiple Linear Regression Line")
plt.xlabel("Total Bill")
plt.ylabel("Tip Amount")


plt.scatter(
    df["total_bill"],
    y,
    color='blue',
    alpha=0.6,
    label='Actual Tips'
)


y_pred = model.predict(
    scaler.transform(df[["total_bill", "size"]])
)

plt.plot(
    df["total_bill"],
    y_pred,
    color='red',
    linewidth=2,
    label='Predicted Tips'
)

plt.legend()
plt.grid(alpha=0.2)

st.pyplot(plt.gcf())
st.markdown('</div>', unsafe_allow_html=True)

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

bill = st.number_input("Total Bill", value=50.0)
size = st.number_input("Table Size", value=2)

if st.button("Predict Tip"):
    tip_pred = model.predict(scaler.transform([[bill, size]]))[0]
    st.markdown(
        f"<div class='prediction-box'>💰 Predicted Tip: <b>${tip_pred:.2f}</b></div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  Multiple Linear Regression Model
</div>
""", unsafe_allow_html=True)
