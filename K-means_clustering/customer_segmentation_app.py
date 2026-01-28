import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #f4f6fb;
}
.header {
    background: linear-gradient(135deg, #43cea2, #185a9d);
    padding: 32px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
}
.header h1 {
    font-size: 40px;
    margin-bottom: 10px;
}
.header p {
    font-size: 18px;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 22px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
.insight {
    background: #eef4ff;
    padding: 18px;
    border-radius: 14px;
    font-size: 16px;
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
    <h1>🟢 Customer Segmentation Dashboard</h1>
    <p>
        This system uses <b>K-Means Clustering</b> to group customers based on their
        purchasing behavior and similarities.
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "Wholesale customers data.csv")

    if not os.path.exists(path):
        st.error("Dataset file not found. Please upload 'Wholesale customers data.csv'")
        st.stop()

    return pd.read_csv(path)

df = load_data()

numerical_features = [
    "Fresh", "Milk", "Grocery",
    "Frozen", "Detergents_Paper", "Delicassen"
]
st.sidebar.header("⚙️ Clustering Controls")

feature_x = st.sidebar.selectbox(
    "Select Feature 1 (X-axis)",
    numerical_features,
    index=0
)

feature_y = st.sidebar.selectbox(
    "Select Feature 2 (Y-axis)",
    numerical_features,
    index=1
)

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=3
)

random_state = st.sidebar.number_input(
    "Random State (Optional)",
    min_value=0,
    value=42
)

run_btn = st.sidebar.button("🟦 Run Clustering")

if run_btn:

    X = df[[feature_x, feature_y]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state
    )

    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Customer Clusters Visualization")

    fig, ax = plt.subplots(figsize=(8,6))

    scatter = ax.scatter(
        df[feature_x],
        df[feature_y],
        c=df["Cluster"],
        cmap="viridis",
        alpha=0.7
    )

    ax.scatter(
        centers[:,0],
        centers[:,1],
        c="red",
        s=250,
        marker="X",
        label="Cluster Centers"
    )

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.legend()

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Cluster Summary")

    summary = df.groupby("Cluster").agg(
        Count=("Cluster", "count"),
        Avg_Feature_1=(feature_x, "mean"),
        Avg_Feature_2=(feature_y, "mean")
    )

    st.dataframe(summary)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💼 Business Interpretation")

    for c in summary.index:
        avg_x = summary.loc[c, "Avg_Feature_1"]
        avg_y = summary.loc[c, "Avg_Feature_2"]

        st.markdown(
            f"**Cluster {c}:** Customers in this group show "
            f"{'high' if avg_x > summary['Avg_Feature_1'].mean() else 'lower'} "
            f"spending in **{feature_x}** and "
            f"{'high' if avg_y > summary['Avg_Feature_2'].mean() else 'lower'} "
            f"spending in **{feature_y}**."
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card insight'>", unsafe_allow_html=True)
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies such as promotions, "
        "pricing, or inventory planning."
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👈 Select features, choose K, and click **Run Clustering** to begin.")

st.markdown("""
<div class="footer">
Customer Segmentation Dashboard using K-Means Clustering
</div>
""", unsafe_allow_html=True)
