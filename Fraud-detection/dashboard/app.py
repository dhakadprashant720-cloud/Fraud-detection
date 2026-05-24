import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🔍", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Fraud-detection/data/sample_results.csv")
    return df

@st.cache_resource
def load_model():
    with open("Fraud-detection/dashboard/model.pkl", "rb") as f:
        model = pickle.load(f)
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

# ==========================================
# PAGE 1 — OVERVIEW
# ==========================================
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
        st.subheader("Risk Tier Distribution")
        tier_counts = df_filtered["RiskTier"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.pie(tier_counts, labels=tier_counts.index, autopct="%1.1f%%",
                colors=["#e74c3c", "#f39c12", "#2ecc71"], wedgeprops={"width": 0.5})
        ax1.set_title("Risk Tier Donut")
        st.pyplot(fig1)
        plt.close()

    with col_b:
        st.subheader("Fraud Probability by Hour")
        hour_data = df_filtered.groupby("HourOfDay")["FraudProbability"].mean()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.bar(hour_data.index, hour_data.values, color="coral")
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Avg Fraud Probability")
        ax2.set_title("Fraud by Hour")
        st.pyplot(fig2)
        plt.close()

    st.subheader("Transaction Amount Distribution")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    for tier, color in [("Critical Risk","#e74c3c"), ("Suspicious","#f39c12"), ("Clear","#2ecc71")]:
        subset = df_filtered[df_filtered["RiskTier"] == tier]["TransactionAmt"]
        ax3.hist(subset, bins=50, alpha=0.6, label=tier, color=color, log=True)
    ax3.set_xlabel("Transaction Amount")
    ax3.set_ylabel("Count (log)")
    ax3.legend()
    st.pyplot(fig3)
    plt.close()

    st.subheader("📋 Risk Tier Summary")
    summary = df_filtered.groupby("RiskTier").agg(
        Count=("TransactionAmt","count"),
        AvgAmt=("TransactionAmt","mean"),
        AvgProb=("FraudProbability","mean")
    ).round(2).reset_index()
    st.dataframe(summary, use_container_width=True)

# ==========================================
# PAGE 2 — TRANSACTION EXPLORER
# ==========================================
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
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sample = df_filtered.sample(min(1000, len(df_filtered)), random_state=42)
    sc = ax4.scatter(sample["HourOfDay"], sample["TransactionAmt"],
                     c=sample["FraudProbability"], cmap="RdYlGn_r", alpha=0.5, s=10)
    plt.colorbar(sc, ax=ax4, label="Fraud Probability")
    ax4.set_xlabel("Hour of Day")
    ax4.set_ylabel("Transaction Amount")
    ax4.set_title("Amount vs Hour — Color = Fraud Probability")
    st.pyplot(fig4)
    plt.close()

# ==========================================
# PAGE 3 — SHAP EXPLAINER
# ==========================================
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
