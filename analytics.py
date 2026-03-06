import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_data(filepath: str = "UniversalBank.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Drop columns we must not use
    df.drop(columns=["ZIP Code", "ID"], errors="ignore", inplace=True)
    # Rename for readability
    df.rename(columns={"Personal Loan": "PersonalLoan",
                        "Securities Account": "SecuritiesAccount",
                        "CD Account": "CDAccount",
                        "CreditCard": "CreditCard"}, inplace=True)
    # Fix any negative experience (data quirk)
    df["Experience"] = df["Experience"].clip(lower=0)
    # Income group
    df["IncomeGroup"] = pd.cut(
        df["Income"],
        bins=[0, 50, 100, 150, 250],
        labels=["Low (<50k)", "Mid (50-100k)", "High (100-150k)", "Very High (>150k)"]
    )
    # Education label
    edu_map = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof"}
    df["EducationLabel"] = df["Education"].map(edu_map)
    # Age group
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60+"]
    )
    return df


def apply_filters(df: pd.DataFrame, income_range, edu_levels, family_sizes) -> pd.DataFrame:
    mask = (
        (df["Income"] >= income_range[0]) &
        (df["Income"] <= income_range[1])
    )
    if edu_levels:
        mask &= df["Education"].isin(edu_levels)
    if family_sizes:
        mask &= df["Family"].isin(family_sizes)
    return df[mask].copy()


# ─────────────────────────────────────────────
# DESCRIPTIVE ANALYTICS
# ─────────────────────────────────────────────

def descriptive_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    accepted = df["PersonalLoan"].sum()
    rejected = total - accepted

    numeric_cols = ["Age", "Income", "CCAvg", "Mortgage", "Experience"]
    summary = df[numeric_cols].describe().T
    summary["median"] = df[numeric_cols].median()

    return {
        "total": total,
        "accepted": int(accepted),
        "rejected": int(rejected),
        "acceptance_rate": round(accepted / total * 100, 2),
        "avg_income": round(df["Income"].mean(), 2),
        "avg_ccavg": round(df["CCAvg"].mean(), 2),
        "avg_mortgage": round(df["Mortgage"].mean(), 2),
        "avg_age": round(df["Age"].mean(), 2),
        "summary_stats": summary,
    }


# ─────────────────────────────────────────────
# DIAGNOSTIC ANALYTICS
# ─────────────────────────────────────────────

def diagnostic_comparison(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Income", "CCAvg", "Mortgage", "Age", "Experience", "Family"]
    grp = df.groupby("PersonalLoan")[cols].mean().T
    grp.columns = ["Rejected (0)", "Accepted (1)"]
    grp["Difference"] = grp["Accepted (1)"] - grp["Rejected (0)"]
    grp["% Change"] = ((grp["Difference"] / grp["Rejected (0)"]) * 100).round(1)
    return grp.round(2)


def acceptance_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    result = df.groupby(col)["PersonalLoan"].agg(
        Total="count",
        Accepted="sum"
    ).reset_index()
    result["Rejected"] = result["Total"] - result["Accepted"]
    result["Acceptance Rate (%)"] = (result["Accepted"] / result["Total"] * 100).round(1)
    return result.sort_values("Acceptance Rate (%)", ascending=False)


def banking_service_analysis(df: pd.DataFrame) -> pd.DataFrame:
    services = ["SecuritiesAccount", "CDAccount", "Online", "CreditCard"]
    rows = []
    for svc in services:
        for has in [0, 1]:
            sub = df[df[svc] == has]
            rate = sub["PersonalLoan"].mean() * 100
            rows.append({
                "Service": svc,
                "Has Service": "Yes" if has == 1 else "No",
                "Count": len(sub),
                "Acceptance Rate (%)": round(rate, 1)
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PREDICTIVE ANALYTICS
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "Age", "Experience", "Income", "Family",
    "CCAvg", "Education", "Mortgage",
    "SecuritiesAccount", "CDAccount", "Online", "CreditCard"
]


def train_models(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0)
    y = df["PersonalLoan"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # Feature importance
    feat_imp = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    # ROC curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)

    return {
        "X_test": X_test,
        "y_test": y_test,
        "lr_pred": lr_pred,
        "lr_proba": lr_proba,
        "lr_report": classification_report(y_test, lr_pred, output_dict=True),
        "lr_cm": confusion_matrix(y_test, lr_pred),
        "lr_auc": round(roc_auc_score(y_test, lr_proba), 4),
        "lr_fpr": lr_fpr,
        "lr_tpr": lr_tpr,
        "rf_pred": rf_pred,
        "rf_proba": rf_proba,
        "rf_report": classification_report(y_test, rf_pred, output_dict=True),
        "rf_cm": confusion_matrix(y_test, rf_pred),
        "rf_auc": round(roc_auc_score(y_test, rf_proba), 4),
        "rf_fpr": rf_fpr,
        "rf_tpr": rf_tpr,
        "feat_imp": feat_imp,
        "scaler": scaler,
        "lr_model": lr,
        "rf_model": rf,
    }


def predict_single(model_results: dict, input_data: dict) -> dict:
    rf = model_results["rf_model"]
    X = pd.DataFrame([input_data])[FEATURE_COLS].fillna(0)
    prob = rf.predict_proba(X)[0][1]
    pred = rf.predict(X)[0]
    return {"probability": round(prob * 100, 1), "prediction": int(pred)}


# ─────────────────────────────────────────────
# PRESCRIPTIVE ANALYTICS
# ─────────────────────────────────────────────

def segment_customers(df: pd.DataFrame) -> pd.DataFrame:
    def classify(row):
        score = 0
        if row["Income"] > 100:
            score += 3
        elif row["Income"] > 50:
            score += 1
        if row["CCAvg"] > 3:
            score += 2
        if row["Education"] == 3:
            score += 2
        elif row["Education"] == 2:
            score += 1
        if row["CDAccount"] == 1:
            score += 2
        if row["Family"] >= 3:
            score += 1
        if row["Mortgage"] > 0:
            score += 1

        if score >= 7:
            return "🔥 Prime Target"
        elif score >= 4:
            return "⚡ High Potential"
        elif score >= 2:
            return "💡 Moderate Potential"
        else:
            return "❄️ Low Priority"

    df = df.copy()
    df["Segment"] = df.apply(classify, axis=1)
    return df


def prescriptive_summary(df_seg: pd.DataFrame) -> pd.DataFrame:
    grp = df_seg.groupby("Segment").agg(
        Count=("PersonalLoan", "count"),
        Loan_Accepted=("PersonalLoan", "sum"),
        Avg_Income=("Income", "mean"),
        Avg_CCAvg=("CCAvg", "mean"),
        Avg_Age=("Age", "mean"),
    ).reset_index()
    grp["Actual Acceptance Rate (%)"] = (
        grp["Loan_Accepted"] / grp["Count"] * 100
    ).round(1)
    grp["Avg_Income"] = grp["Avg_Income"].round(1)
    grp["Avg_CCAvg"] = grp["Avg_CCAvg"].round(2)
    grp["Avg_Age"] = grp["Avg_Age"].round(1)
    return grp.sort_values("Actual Acceptance Rate (%)", ascending=False)
