# 🏦 Universal Bank – Personal Loan Analytics Dashboard

A comprehensive **4-type analytics dashboard** built with Streamlit to identify which customers are most likely to accept a personal loan offer.

## 📊 Dashboard Sections

| Tab | Analytics Type | What it covers |
|-----|---------------|----------------|
| 📊 Descriptive | What happened | Demographics, loan acceptance rates, income & CC spending distributions |
| 🔍 Diagnostic | Why it happened | Group comparisons, banking service analysis, correlation heatmap, interactive drill-downs |
| 🤖 Predictive | What will happen | Random Forest + Logistic Regression models, ROC curves, live customer predictor |
| 🎯 Prescriptive | What to do | Customer segmentation, targeting strategy, campaign ROI estimates |

## 🚀 Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo and set **Main file path** to `app.py`
5. Click **Deploy** — done!

## 📁 File Structure

```
universal_bank_dashboard/
├── app.py              # Main Streamlit dashboard
├── analytics.py        # Analytics & ML helper module
├── UniversalBank.csv   # Dataset (5,000 customers)
├── requirements.txt    # Python dependencies
└── README.md
```

## 🎛️ Dashboard Features

- **Sidebar Filters**: Income range, education level, family size
- **KPI Cards**: Total customers, acceptance rate, avg income/spending/mortgage
- **Interactive Charts**: Sunburst, Treemap drill-downs, scatter, box plots, heatmap
- **ML Models**: Logistic Regression & Random Forest with AUC, confusion matrices, feature importance
- **Live Predictor**: Enter any customer profile → get loan acceptance probability
- **Prescriptive Segments**: 4-tier customer segmentation with strategic recommendations

## 🔑 Key Findings

- **Income** is the #1 predictor of loan acceptance
- **CD Account holders** accept loans at 3× the average rate
- **Advanced/Professional** education customers show highest acceptance
- **Credit card spending** (CCAvg) strongly correlates with loan acceptance
- Random Forest achieves **AUC ~0.98** on test set

## 📦 Dependencies

- `streamlit` – Dashboard framework  
- `plotly` – Interactive visualisations  
- `scikit-learn` – Machine learning models  
- `pandas` / `numpy` – Data processing  

---
*Target variable: `Personal Loan` (0 = Rejected, 1 = Accepted)*
