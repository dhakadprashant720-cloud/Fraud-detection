
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🔍", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/sample_results.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("dashboard/model.pkl")
    return model

df    = load_data()
model = load_model()

st.sidebar.title("🔍 Fraud Detection")
st.sidebar.markdown("**By: Prashant Dhakad**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate to", ["📊 Overview", "🔎 Transaction Explorer", "🧠 SHAP Explainer"])

risk_filter  = st.sidebar.multiselect("Risk Tier", ["Critical Risk", "Suspicious", "Clear"], default=["Critical Risk", "Suspicious", "Clear"])
amount_range = st.sidebar.slider("Transaction Amount ($)", 0, int(df["TransactionAmt"].max()), (0, int(df["TransactionAmt"].max())))

df_filtered = df[df["RiskTier"].isin(risk_filter) & df["TransactionAmt"].between(amount_range[0], amount_range[1])]

if page == "📊 Overview":
    st.title("📊 Fraud Detection — Overview Dashboard")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    total_tx       = len(df_filtered)
    total_fraud    = int((df_filtered["FraudProbability"] >= 0.5).sum())
    detection_rate = round(total_fraud / total_tx * 100, 2) if total_tx > 0 else 0
    avg_fraud_amt  = round(df_filtered[df_filtered["FraudProbability"] >= 0.5]["TransactionAmt"].mean(), 2) if total_fraud > 0 else 0

    col1.metric("🧾 Total Transactions", f"{total_tx:,}")
    col2.metric("🚨 Total Fraud Count",  f"{total_fraud:,}")
    col3.metric("📈 Detection Rate",     f"{detection_rate}%")
    col4.metric("💰 Avg Fraud Amount",   f"${avg_fraud_amt}")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        tier_counts = df_filtered["RiskTier"].value_counts()
        fig = go.Figure(go.Pie(labels=tier_counts.index, values=tier_counts.values, hole=0.5, marker_colors=["#e74c3c","#f39c12","#2ecc71"]))
        fig.update_layout(title="Risk Tier Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        hour_data = df_filtered.groupby("HourOfDay")["FraudProbability"].mean().reset_index()
        fig2 = px.bar(hour_data, x="HourOfDay", y="FraudProbability", title="Fraud Probability by Hour", color="FraudProbability", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(df_filtered, x="TransactionAmt", color="RiskTier", nbins=50, log_y=True, title="Transaction Amount Distribution", color_discrete_map={"Critical Risk":"#e74c3c","Suspicious":"#f39c12","Clear":"#2ecc71"})
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📋 Risk Tier Summary")
    summary = df_filtered.groupby("RiskTier").agg(Count=("TransactionAmt","count"), AvgAmt=("TransactionAmt","mean"), AvgProb=("FraudProbability","mean")).round(2).reset_index()
    st.dataframe(summary, use_container_width=True)

elif page == "🔎 Transaction Explorer":
    st.title("🔎 Transaction Explorer")
    st.markdown("---")

    search_id = st.text_input("🔍 Search by Transaction ID", placeholder="e.g. 3000001")
    if search_id:
        try:
            result = df[df["TransactionID"] == int(search_id)]
            if len(result) > 0:
                prob = result["FraudProbability"].values[0]
                risk = result["RiskTier"].values[0]
                amt  = result["TransactionAmt"].values[0]
                hour = result["HourOfDay"].values[0]
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Risk Tier", risk)
                c2.metric("Fraud Probability", f"{prob:.4f}")
                c3.metric("Amount", f"${amt}")
                c4.metric("Hour", f"{int(hour)}:00")
                if risk == "Critical Risk":
                    st.error("🔴 HIGH RISK — Block transaction and flag for review")
                elif risk == "Suspicious":
                    st.warning("🟡 SUSPICIOUS — Send OTP verification")
                else:
                    st.success("🟢 CLEAR — Transaction looks normal")
            else:
                st.warning("Transaction ID not found.")
        except:
            st.error("Please enter a valid numeric Transaction ID.")

    st.markdown("---")
    st.subheader("📋 All Transactions")
    sort_col = st.selectbox("Sort By", ["FraudProbability","TransactionAmt","HourOfDay"])
    sort_asc = st.checkbox("Ascending", value=False)
    st.dataframe(df_filtered.sort_values(sort_col, ascending=sort_asc).head(200), use_container_width=True)

    fig4 = px.scatter(df_filtered.sample(min(1000,len(df_filtered)),random_state=42), x="HourOfDay", y="TransactionAmt", color="FraudProbability", color_continuous_scale="RdYlGn_r", title="Amount vs Hour — Color = Fraud Probability", opacity=0.6)
    st.plotly_chart(fig4, use_container_width=True)

elif page == "🧠 SHAP Explainer":
    st.title("🧠 SHAP Explainer")
    st.markdown("---")

    tx_id = st.text_input("Enter Transaction ID", placeholder="e.g. 3000001")
    if tx_id:
        try:
            result = df[df["TransactionID"] == int(tx_id)]
            if len(result) > 0:
                prob = result["FraudProbability"].values[0]
                risk = result["RiskTier"].values[0]
                amt  = result["TransactionAmt"].values[0]
                hour = result["HourOfDay"].values[0]
                c1,c2,c3 = st.columns(3)
                c1.metric("Fraud Probability", f"{prob:.4f}")
                c2.metric("Amount", f"${amt}")
                c3.metric("Hour", f"{int(hour)}:00")
                st.markdown("---")
                if risk == "Critical Risk":
                    st.error("🔴 CRITICAL RISK")
                    st.markdown(f"Is transaction mein fraud probability bahut zyada hai ({prob:.2%}). Amount ${amt} suspicious hai aur time {int(hour)}:00 unusual hai. **Action: Block karo.**")
                elif risk == "Suspicious":
                    st.warning("🟡 SUSPICIOUS")
                    st.markdown(f"Fraud probability medium hai ({prob:.2%}). Kuch features suspicious hain. **Action: OTP verification bhejo.**")
                else:
                    st.success("🟢 CLEAR")
                    st.markdown(f"Fraud probability bahut kam hai ({prob:.2%}). Transaction bilkul normal lagti hai. **Action: Allow karo.**")
            else:
                st.warning("Transaction ID not found.")
        except:
            st.error("Valid numeric Transaction ID daalo.")

    st.markdown("---")
    st.subheader("ℹ️ Model Info")
    st.markdown("""
    | Step | Description |
    |------|-------------|
    | Model | LightGBM (best performer) |
    | Imbalance | SMOTE applied |
    | Features | 3 engineered features |
    | Explainability | SHAP values |
    """)
