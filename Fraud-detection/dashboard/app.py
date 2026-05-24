import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🔍", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Fraud-detection/data/sample_results.csv")
    return df

df = load_data()

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

    st.subheader("Risk Tier Distribution")
    tier_counts = df_filtered["RiskTier"].value_counts().reset_index()
    tier_counts.columns = ["RiskTier", "Count"]
    st.bar_chart(tier_counts.set_index("RiskTier"))

    st.subheader("Fraud Probability by Hour of Day")
    hour_data = df_filtered.groupby("HourOfDay")["FraudProbability"].mean()
    st.bar_chart(hour_data)

    st.subheader("Transaction Amount Distribution")
    amt_data = df_filtered.groupby(pd.cut(df_filtered["TransactionAmt"], bins=20))["TransactionAmt"].count()
    amt_df = df_filtered[["TransactionAmt"]].copy()
    amt_df["bin"] = pd.cut(amt_df["TransactionAmt"], bins=20).astype(str)
    amt_counts = amt_df.groupby("bin")["TransactionAmt"].count().reset_index()
    amt_counts.columns = ["Amount Range", "Count"]
    st.bar_chart(amt_counts.set_index("Amount Range"))

    st.subheader("📋 Risk Tier Summary")
    summary = df_filtered.groupby("RiskTier").agg(
        Count=("TransactionAmt","count"),
        AvgAmt=("TransactionAmt","mean"),
        AvgProb=("FraudProbability","mean")
    ).round(2).reset_index()
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
                c1, c2, c3, c4 = st.columns(4)
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
    sort_col = st.selectbox("Sort By", ["FraudProbability", "TransactionAmt", "HourOfDay"])
    sort_asc = st.checkbox("Ascending", value=False)
    st.dataframe(df_filtered.sort_values(sort_col, ascending=sort_asc).head(200), use_container_width=True)

    st.subheader("Amount vs Hour of Day")
    chart_data = df_filtered[["HourOfDay", "TransactionAmt", "FraudProbability"]].sample(min(500, len(df_filtered)), random_state=42)
    st.scatter_chart(chart_data, x="HourOfDay", y="TransactionAmt", color="FraudProbability")

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
                c1, c2, c3 = st.columns(3)
                c1.metric("Fraud Probability", f"{prob:.4f}")
                c2.metric("Amount", f"${amt}")
                c3.metric("Hour", f"{int(hour)}:00")
                st.markdown("---")
                if risk == "Critical Risk":
                    st.error("🔴 CRITICAL RISK")
                    st.markdown(f"Fraud probability bahut zyada hai ({prob:.2%}). Amount ${amt} suspicious hai, time {int(hour)}:00 unusual hai. **Action: Block karo.**")
                elif risk == "Suspicious":
                    st.warning("🟡 SUSPICIOUS")
                    st.markdown(f"Fraud probability medium hai ({prob:.2%}). Kuch features suspicious hain. **Action: OTP verification bhejo.**")
                else:
                    st.success("🟢 CLEAR")
                    st.markdown(f"Fraud probability bahut kam hai ({prob:.2%}). Transaction normal lagti hai. **Action: Allow karo.**")
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
