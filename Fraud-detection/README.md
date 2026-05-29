#  Real-Time Fraud Detection System
### By: Prashant Dhakad | AI & Data Analytics 

---

##  Folder Structure

```
Fraud_detection_prashantdhakad/
│
├── data/
│   ├── train_transaction.csv         ← download from Kaggle
│   ├── train_identity.csv            ← download from Kaggle
│   └── sample_results.csv            ← notebook run karne ke baad banta hai
│
├── dashboard/
│   ├── app.py                        ← Streamlit dashboard (3 pages)
│   └── model.pkl                     ← Trained LightGBM model
│
├── Models/
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   ├── isolation_forest_model.pkl
│   └── scaler.pkl
│
├── charts/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── threshold_f1.png
│   ├── fraud_by_hour.png
│   ├── amt_distribution.png
│   ├── correlation_heatmap.png
│   ├── risk_tier_donut.png
│   ├── risk_tier_comparison.png
│   ├── pr_with_threshold.png
│   ├── scatter_bonus.png
│   ├── shap_waterfall_fraud.png
│   ├── shap_waterfall_border.png
│   ├── shap_waterfall_legit.png
│   ├── shap_dependence.png
│   └── shap_vs_model_importance.png
│
├── analysis.ipynb           ← MAIN NOTEBOOK (Task 1 to Task 8)
├── model_comparison.png
├── shap_summary.png
├── summary.docx                      ← Project Summary Report
├── requirements.txt
└── README.md
```

---

##  Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Data Loading, Merging & EDA | 
| Task 2 | Preprocessing, SMOTE, Feature Engineering | 
| Task 3 | LightGBM + XGBoost + Isolation Forest + Threshold Optimization | 
| Task 4 | Explainable AI with SHAP Values | 
| Task 5 | Risk Segmentation (Critical / Suspicious / Clear) | 
| Task 6 | Streamlit Dashboard — 3 pages | 
| Task 7 | 5+ Charts saved in charts/ folder | 
| Task 8 | Business Insights & Recommendations | 

---

##  How to Run

### Step 1 — Dataset download karo
1. Kaggle.com pe jao → "IEEE-CIS Fraud Detection" search 
2. `train_transaction.csv` aur `train_identity.csv` download 
3. Drive mein `Fraud_detection_prashantdhakad/data/`put in folder 

### Step 2 — Notebook run karo (Google Colab)
1. `analysis_complete.ipynb`open in Colab 
2. **Runtime > Run All** 
3. All charts, models, CSV automatically Drive mein save ho jayenge

### Step 3 — Dashboard locally
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

##  Live Dashboard
**https://fraud-detection-bvzktssw6zvmqxnm5ezabm.streamlit.app/**

---

##  Key Results

| Model | ROC-AUC | F1 Score |
|-------|---------|----------|
| LightGBM (Best) | ~0.95 | ~0.81 |
| XGBoost | ~0.93 | ~0.78 |
| Isolation Forest | ~0.65 | ~0.42 |

**Best Model: LightGBM** — Highest ROC-AUC and F1 Score

---

##  Top Fraud Signals (SHAP)
1. **TransactionAmt** — High amounts = biggest fraud signal
2. **card1 / card features** — Certain card patterns linked to fraud
3. **HourOfDay** — Late night transactions (1AM-5AM) are riskier

---

## 🛠 Tools Used
Python | Pandas | NumPy | LightGBM | XGBoost | Scikit-learn | SMOTE | SHAP | Streamlit | Google Colab | GitHub
