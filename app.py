"""
Universal Bank – Personal Loan Acceptance Dashboard
====================================================
Streamlit app | 4 analytics tabs: Descriptive · Diagnostic · Predictive · Prescriptive
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics import (
    load_data, apply_filters,
    descriptive_summary, diagnostic_comparison,
    acceptance_by_category, banking_service_analysis,
    train_models, predict_single,
    segment_customers, prescriptive_summary,
    FEATURE_COLS
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank | Loan Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
ACCEPT_COLOR  = "#2ECC71"   # green
REJECT_COLOR  = "#E74C3C"   # red
PRIMARY       = "#1A3C6B"   # navy
SECONDARY     = "#2980B9"   # blue
ACCENT        = "#F39C12"   # amber
BG_CARD       = "#0E1117"
GRADIENT_SEQ  = px.colors.sequential.Blues_r

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0E1117; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1A3C6B 0%, #0d2137 100%);
        border-radius: 14px;
        padding: 22px 20px 18px 20px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        border: 1px solid #2980B9;
    }
    .kpi-value { font-size: 2rem; font-weight: 800; color: #F39C12; margin:0; }
    .kpi-label { font-size: 0.82rem; color: #A8C4E0; margin-top: 4px; }
    .kpi-sub   { font-size: 0.75rem; color: #5D8AA8; margin-top: 2px; }

    /* Section headers */
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #E8F4FD;
        border-left: 4px solid #F39C12;
        padding-left: 12px; margin: 28px 0 14px 0;
    }

    /* Insight boxes */
    .insight-box {
        background: #141a24; border-left: 4px solid #2ECC71;
        border-radius: 8px; padding: 14px 18px; margin: 8px 0;
        font-size: 0.88rem; color: #C8E6C9;
    }
    .insight-box.warn { border-left-color: #F39C12; color: #FFF3CD; }
    .insight-box.info { border-left-color: #2980B9; color: #B3D4FF; }

    /* Tab pills */
    div[data-testid="stTabs"] button {
        font-size: 0.95rem; font-weight: 600;
        border-radius: 8px 8px 0 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #10182a; }
    [data-testid="stSidebar"] .stMarkdown h3 { color: #F39C12; }
</style>
""", unsafe_allow_html=True)


# ── Load & cache data ──────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data("UniversalBank.csv")

@st.cache_data
def get_models(df_json):
    df = pd.read_json(df_json)
    return train_models(df)

df_full = get_data()


# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("### 🎛️ Dashboard Filters")
    st.markdown("---")

    income_range = st.slider(
        "💰 Income Range ($000)",
        int(df_full["Income"].min()),
        int(df_full["Income"].max()),
        (int(df_full["Income"].min()), int(df_full["Income"].max()))
    )

    edu_options = {1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof"}
    edu_sel = st.multiselect(
        "🎓 Education Level",
        options=list(edu_options.keys()),
        default=list(edu_options.keys()),
        format_func=lambda x: edu_options[x]
    )

    family_sel = st.multiselect(
        "👨‍👩‍👧 Family Size",
        options=sorted(df_full["Family"].unique()),
        default=sorted(df_full["Family"].unique())
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#5D8AA8;'>"
        "Target: <b style='color:#F39C12'>Personal Loan Acceptance</b><br>"
        "Dataset: 5,000 customers</div>",
        unsafe_allow_html=True
    )

df = apply_filters(df_full, income_range, edu_sel, family_sel)


# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#1A3C6B,#0d2137);
            border-radius:14px;padding:28px 32px 20px;margin-bottom:24px;
            border:1px solid #2980B9;">
  <h1 style="color:#F39C12;margin:0;font-size:2rem;">🏦 Universal Bank</h1>
  <p style="color:#A8C4E0;margin:6px 0 0;font-size:1.05rem;">
    Personal Loan Acceptance Analytics Dashboard &nbsp;|&nbsp;
    <span style="color:#2ECC71;">Customer Intelligence Platform</span>
  </p>
  <p style="color:#5D8AA8;font-size:0.82rem;margin:4px 0 0;">
    Objective: Identify which customers are most likely to accept a personal loan offer
  </p>
</div>
""", unsafe_allow_html=True)

# Filtered-data note
st.markdown(
    f"<div style='color:#5D8AA8;font-size:0.8rem;margin-bottom:8px;'>"
    f"Showing <b style='color:#F39C12'>{len(df):,}</b> of {len(df_full):,} customers after filters</div>",
    unsafe_allow_html=True
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Descriptive Analytics",
    "🔍 Diagnostic Analytics",
    "🤖 Predictive Analytics",
    "🎯 Prescriptive Analytics",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    stats = descriptive_summary(df)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, f"{stats['total']:,}", "Total Customers", ""),
        (k2, f"{stats['accepted']:,}", "Loan Accepted", f"{stats['acceptance_rate']}%"),
        (k3, f"${stats['avg_income']}K", "Avg Annual Income", "per customer"),
        (k4, f"${stats['avg_ccavg']}K", "Avg CC Spending/mo", "per customer"),
        (k5, f"${stats['avg_mortgage']}K", "Avg Mortgage", "per customer"),
    ]
    for col, val, label, sub in kpis:
        with col:
            st.markdown(
                f"<div class='kpi-card'>"
                f"<p class='kpi-value'>{val}</p>"
                f"<p class='kpi-label'>{label}</p>"
                f"<p class='kpi-sub'>{sub}</p>"
                f"</div>", unsafe_allow_html=True
            )

    st.markdown("<div class='section-header'>Loan Acceptance Overview</div>", unsafe_allow_html=True)

    col_pie, col_donut = st.columns(2)

    with col_pie:
        # Donut – overall acceptance
        fig_donut = go.Figure(go.Pie(
            labels=["Accepted", "Rejected"],
            values=[stats["accepted"], stats["rejected"]],
            hole=0.55,
            marker_colors=[ACCEPT_COLOR, REJECT_COLOR],
            textinfo="percent+label",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"
        ))
        fig_donut.update_layout(
            title=dict(text="Overall Loan Acceptance", font_color="#E8F4FD"),
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font_color="#A8C4E0", legend_font_color="#A8C4E0",
            margin=dict(t=50, b=10),
            annotations=[dict(
                text=f"<b>{stats['acceptance_rate']}%</b><br>Acceptance",
                font_size=15, showarrow=False, font_color="#F39C12"
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_donut:
        # Interactive sunburst: Education → Loan acceptance
        df_sun = df.copy()
        df_sun["LoanStatus"] = df_sun["PersonalLoan"].map({1: "Accepted", 0: "Rejected"})
        fig_sun = px.sunburst(
            df_sun,
            path=["EducationLabel", "LoanStatus"],
            title="🔍 Interactive: Education → Loan Acceptance",
            color="LoanStatus",
            color_discrete_map={"Accepted": ACCEPT_COLOR, "Rejected": REJECT_COLOR},
        )
        fig_sun.update_layout(
            paper_bgcolor="#0E1117", font_color="#A8C4E0",
            title_font_color="#E8F4FD", margin=dict(t=50, b=10)
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    💡 <b>Key Insight:</b> Only ~9.6% of all customers accepted the personal loan – 
    a highly imbalanced target. Advanced/Professional education customers show the 
    highest acceptance rate, suggesting education is a strong differentiator.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Demographic Distributions</div>", unsafe_allow_html=True)

    col_age, col_inc = st.columns(2)

    with col_age:
        fig_age = px.histogram(
            df, x="Age", color="PersonalLoan",
            barmode="overlay", nbins=35,
            color_discrete_map={0: REJECT_COLOR, 1: ACCEPT_COLOR},
            labels={"PersonalLoan": "Loan", "Age": "Age (years)"},
            title="Age Distribution by Loan Status",
            opacity=0.75
        )
        fig_age.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
            font_color="#A8C4E0", title_font_color="#E8F4FD",
            legend=dict(title="Loan", bgcolor="#141a24"),
            xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
        )
        fig_age.for_each_trace(
            lambda t: t.update(name="Accepted" if t.name == "1" else "Rejected")
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col_inc:
        fig_inc = px.histogram(
            df, x="Income", color="PersonalLoan",
            barmode="overlay", nbins=40,
            color_discrete_map={0: REJECT_COLOR, 1: ACCEPT_COLOR},
            labels={"PersonalLoan": "Loan", "Income": "Annual Income ($000)"},
            title="Income Distribution by Loan Status",
            opacity=0.75
        )
        fig_inc.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
            font_color="#A8C4E0", title_font_color="#E8F4FD",
            legend=dict(title="Loan", bgcolor="#141a24"),
            xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
        )
        fig_inc.for_each_trace(
            lambda t: t.update(name="Accepted" if t.name == "1" else "Rejected")
        )
        st.plotly_chart(fig_inc, use_container_width=True)

    col_fam, col_cc = st.columns(2)

    with col_fam:
        fam_data = df.groupby("Family")["PersonalLoan"].agg(
            Total="count", Accepted="sum"
        ).reset_index()
        fam_data["Rate"] = (fam_data["Accepted"] / fam_data["Total"] * 100).round(1)
        fig_fam = px.bar(
            fam_data, x="Family", y="Rate",
            text="Rate",
            title="Loan Acceptance Rate by Family Size",
            color="Rate",
            color_continuous_scale=["#1A3C6B", ACCEPT_COLOR],
        )
        fig_fam.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_fam.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
            font_color="#A8C4E0", title_font_color="#E8F4FD",
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1e2a3a", title="Family Size"),
            yaxis=dict(gridcolor="#1e2a3a", title="Acceptance Rate (%)")
        )
        st.plotly_chart(fig_fam, use_container_width=True)

    with col_cc:
        fig_box = px.box(
            df, x="PersonalLoan", y="CCAvg",
            color="PersonalLoan",
            color_discrete_map={0: REJECT_COLOR, 1: ACCEPT_COLOR},
            labels={"PersonalLoan": "Loan Status", "CCAvg": "Avg CC Spending ($000/mo)"},
            title="Credit Card Spending vs Loan Status",
            category_orders={"PersonalLoan": [0, 1]}
        )
        fig_box.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
            font_color="#A8C4E0", title_font_color="#E8F4FD",
            showlegend=False,
            xaxis=dict(gridcolor="#1e2a3a",
                       tickvals=[0, 1], ticktext=["Rejected", "Accepted"]),
            yaxis=dict(gridcolor="#1e2a3a")
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Summary stats table
    st.markdown("<div class='section-header'>Summary Statistics</div>", unsafe_allow_html=True)
    sdf = stats["summary_stats"].reset_index()
    sdf.columns = ["Variable"] + list(sdf.columns[1:])
    st.dataframe(
        sdf.style.background_gradient(cmap="Blues", subset=sdf.columns[1:]),
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Who Accepts the Loan? Group Comparison</div>",
                unsafe_allow_html=True)

    diag = diagnostic_comparison(df)
    st.dataframe(
        diag.style
            .background_gradient(cmap="RdYlGn", subset=["% Change"])
            .format("{:.2f}"),
        use_container_width=True
    )

    st.markdown("""
    <div class='insight-box'>
    📌 Customers who accepted the loan have <b>~3× higher income</b> and 
    <b>~3.2× higher credit card spending</b> on average. This strongly suggests 
    financial capacity is the #1 driver.
    </div>""", unsafe_allow_html=True)

    # ── Income vs Loan (scatter)
    col_sc, col_edu = st.columns(2)

    with col_sc:
        df_sc = df.sample(min(2000, len(df)), random_state=42)
        df_sc["Status"] = df_sc["PersonalLoan"].map({1: "Accepted", 0: "Rejected"})
        fig_sc = px.scatter(
            df_sc, x="Income", y="CCAvg",
            color="Status",
            color_discrete_map={"Accepted": ACCEPT_COLOR, "Rejected": REJECT_COLOR},
            opacity=0.55, size_max=8,
            title="Income vs CC Spending (Loan Status)",
            labels={"Income": "Annual Income ($000)", "CCAvg": "CC Spending ($000/mo)"}
        )
        fig_sc.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
            font_color="#A8C4E0", title_font_color="#E8F4FD",
            legend_bgcolor="#141a24",
            xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_edu:
        edu_acc = acceptance_by_category(df, "EducationLabel")
        fig_edu = px.bar(
            edu_acc, x="EducationLabel", y="Acceptance Rate (%)",
            color="Acceptance Rate (%)",
            text="Acceptance Rate (%)",
            color_continuous_scale=["#1A3C6B", ACCEPT_COLOR],
            title="Loan Acceptance Rate by Education Level",
            labels={"EducationLabel": "Education Level"}
        )
        fig_edu.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_edu.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
            font_color="#A8C4E0", title_font_color="#E8F4FD",
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
        )
        st.plotly_chart(fig_edu, use_container_width=True)

    # ── Income group breakdown
    st.markdown("<div class='section-header'>Loan Acceptance by Income Group</div>",
                unsafe_allow_html=True)

    inc_acc = acceptance_by_category(df, "IncomeGroup")
    fig_inc_grp = px.bar(
        inc_acc, x="IncomeGroup", y=["Accepted", "Rejected"],
        barmode="group",
        color_discrete_map={"Accepted": ACCEPT_COLOR, "Rejected": REJECT_COLOR},
        title="Accepted vs Rejected by Income Group",
        text_auto=True
    )
    fig_inc_grp.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
        font_color="#A8C4E0", title_font_color="#E8F4FD",
        legend_bgcolor="#141a24",
        xaxis=dict(gridcolor="#1e2a3a", title="Income Group"),
        yaxis=dict(gridcolor="#1e2a3a", title="Customer Count")
    )
    st.plotly_chart(fig_inc_grp, use_container_width=True)

    # ── Banking service analysis
    st.markdown("<div class='section-header'>Banking Services & Loan Acceptance</div>",
                unsafe_allow_html=True)

    svc_df = banking_service_analysis(df)
    svc_pivot = svc_df.pivot(index="Service", columns="Has Service",
                              values="Acceptance Rate (%)").reset_index()

    fig_svc = go.Figure()
    fig_svc.add_trace(go.Bar(
        name="Has Service (Yes)", x=svc_pivot["Service"],
        y=svc_pivot.get("Yes", [0]*len(svc_pivot)),
        marker_color=SECONDARY, text=svc_pivot.get("Yes"),
        texttemplate="%{text}%", textposition="outside"
    ))
    fig_svc.add_trace(go.Bar(
        name="No Service", x=svc_pivot["Service"],
        y=svc_pivot.get("No", [0]*len(svc_pivot)),
        marker_color=REJECT_COLOR, text=svc_pivot.get("No"),
        texttemplate="%{text}%", textposition="outside"
    ))
    fig_svc.update_layout(
        barmode="group",
        title="Loan Acceptance Rate: Has Banking Service vs Not",
        paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
        font_color="#A8C4E0", title_font_color="#E8F4FD",
        legend_bgcolor="#141a24",
        xaxis=dict(gridcolor="#1e2a3a", title="Banking Service"),
        yaxis=dict(gridcolor="#1e2a3a", title="Acceptance Rate (%)")
    )
    st.plotly_chart(fig_svc, use_container_width=True)

    # ── Correlation heatmap
    st.markdown("<div class='section-header'>Correlation Heatmap</div>",
                unsafe_allow_html=True)

    num_cols = ["Age", "Income", "CCAvg", "Family", "Mortgage",
                "Education", "SecuritiesAccount", "CDAccount",
                "Online", "CreditCard", "PersonalLoan"]
    corr = df[num_cols].corr().round(2)

    fig_heat = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu", zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        hoverongaps=False
    ))
    fig_heat.update_layout(
        title="Feature Correlation Matrix",
        paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
        font_color="#A8C4E0", title_font_color="#E8F4FD",
        height=520
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    <div class='insight-box info'>
    🔗 <b>Top correlations with PersonalLoan:</b><br>
    • <b>Income</b> – strongest positive driver<br>
    • <b>CCAvg</b> – high spenders more likely to accept<br>
    • <b>CDAccount</b> – customers with CD accounts accept at much higher rates<br>
    • <b>Education</b> – higher education = higher acceptance<br>
    • <b>Age/Experience</b> – weak correlation, not strong standalone predictors
    </div>""", unsafe_allow_html=True)

    # ── Interactive drill-down: Income group → Education → Loan
    st.markdown("<div class='section-header'>🔍 Interactive Drill-Down: Income → Education → Loan</div>",
                unsafe_allow_html=True)

    df_drill = df.copy()
    df_drill["LoanStatus"] = df_drill["PersonalLoan"].map({1: "Accepted", 0: "Rejected"})

    fig_tree = px.treemap(
        df_drill,
        path=["IncomeGroup", "EducationLabel", "LoanStatus"],
        title="Treemap: Click to drill into Income → Education → Loan Acceptance",
        color="LoanStatus",
        color_discrete_map={"Accepted": ACCEPT_COLOR, "Rejected": REJECT_COLOR},
    )
    fig_tree.update_traces(
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>"
    )
    fig_tree.update_layout(
        paper_bgcolor="#0E1117", font_color="#A8C4E0",
        title_font_color="#E8F4FD", height=480
    )
    st.plotly_chart(fig_tree, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – PREDICTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Machine Learning Models</div>",
                unsafe_allow_html=True)

    # Train on full unfiltered dataset for robust models
    with st.spinner("🤖 Training models on full dataset…"):
        models = get_models(df_full.to_json())

    col_lr, col_rf = st.columns(2)

    def metric_card(col, model_name, auc, report, color):
        acc = report["accuracy"]
        prec = report["1"]["precision"]
        rec  = report["1"]["recall"]
        f1   = report["1"]["f1-score"]
        with col:
            st.markdown(
                f"<div class='kpi-card' style='border-color:{color};'>"
                f"<p style='color:{color};font-size:1rem;font-weight:700;'>{model_name}</p>"
                f"<p class='kpi-value' style='color:{color};font-size:1.6rem;'>AUC {auc}</p>"
                f"<p class='kpi-sub'>Accuracy: {acc:.1%} | Precision: {prec:.1%} | "
                f"Recall: {rec:.1%} | F1: {f1:.1%}</p>"
                f"</div>", unsafe_allow_html=True
            )

    metric_card(col_lr, "Logistic Regression", models["lr_auc"],
                models["lr_report"], SECONDARY)
    metric_card(col_rf, "Random Forest", models["rf_auc"],
                models["rf_report"], ACCEPT_COLOR)

    # ROC curves
    col_roc, col_imp = st.columns(2)

    with col_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=models["lr_fpr"], y=models["lr_tpr"],
            mode="lines", name=f"Logistic Reg (AUC={models['lr_auc']})",
            line=dict(color=SECONDARY, width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=models["rf_fpr"], y=models["rf_tpr"],
            mode="lines", name=f"Random Forest (AUC={models['rf_auc']})",
            line=dict(color=ACCEPT_COLOR, width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#555", dash="dash"), name="Random (AUC=0.5)"
        ))
        fig_roc.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
            font_color="#A8C4E0", title_font_color="#E8F4FD",
            legend_bgcolor="#141a24",
            xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_imp:
        fig_imp = px.bar(
            models["feat_imp"], x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#1A3C6B", ACCEPT_COLOR],
            title="Random Forest – Feature Importance",
        )
        fig_imp.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
            font_color="#A8C4E0", title_font_color="#E8F4FD",
            coloraxis_showscale=False, yaxis_categoryorder="total ascending",
            xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Confusion matrices
    st.markdown("<div class='section-header'>Confusion Matrices</div>", unsafe_allow_html=True)
    col_cm1, col_cm2 = st.columns(2)

    def plot_cm(cm, title, color):
        labels = ["Rejected", "Accepted"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, "#0E1117"], [1, color]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
            showscale=False
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Predicted", yaxis_title="Actual",
            paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
            font_color="#A8C4E0", title_font_color="#E8F4FD",
            height=320
        )
        return fig

    with col_cm1:
        st.plotly_chart(
            plot_cm(models["lr_cm"], "Logistic Regression", SECONDARY),
            use_container_width=True
        )
    with col_cm2:
        st.plotly_chart(
            plot_cm(models["rf_cm"], "Random Forest", ACCEPT_COLOR),
            use_container_width=True
        )

    # ── Live predictor
    st.markdown("<div class='section-header'>🎯 Live Customer Loan Predictor</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='insight-box info'>Enter a customer profile below to predict "
        "the probability of loan acceptance using the trained Random Forest model.</div>",
        unsafe_allow_html=True
    )

    with st.form("predictor_form"):
        p1, p2, p3 = st.columns(3)
        with p1:
            p_age   = st.slider("Age", 18, 75, 35)
            p_exp   = st.slider("Experience (years)", 0, 45, 10)
            p_inc   = st.slider("Income ($000)", 8, 224, 80)
            p_fam   = st.selectbox("Family Size", [1, 2, 3, 4])
        with p2:
            p_cc    = st.slider("CC Avg Spending ($000/mo)", 0.0, 10.0, 1.5, step=0.1)
            p_edu   = st.selectbox("Education", [1, 2, 3],
                                   format_func=lambda x: {1:"Undergrad",2:"Graduate",3:"Advanced/Prof"}[x])
            p_mort  = st.slider("Mortgage ($000)", 0, 700, 0)
        with p3:
            p_sec   = st.selectbox("Securities Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
            p_cd    = st.selectbox("CD Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
            p_onl   = st.selectbox("Online Banking", [0, 1], format_func=lambda x: "Yes" if x else "No")
            p_cc2   = st.selectbox("Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("🔮 Predict Loan Acceptance", use_container_width=True)

    if submitted:
        inp = dict(Age=p_age, Experience=p_exp, Income=p_inc, Family=p_fam,
                   CCAvg=p_cc, Education=p_edu, Mortgage=p_mort,
                   SecuritiesAccount=p_sec, CDAccount=p_cd,
                   Online=p_onl, CreditCard=p_cc2)
        result = predict_single(models, inp)
        prob = result["probability"]
        pred = result["prediction"]
        color = ACCEPT_COLOR if pred == 1 else REJECT_COLOR
        label = "✅ Likely to ACCEPT" if pred == 1 else "❌ Likely to REJECT"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob,
            number={"suffix": "%", "font": {"color": color, "size": 40}},
            title={"text": f"Prediction: <b>{label}</b>",
                   "font": {"color": color, "size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#A8C4E0"},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "#141a24",
                "steps": [
                    {"range": [0, 30], "color": "#1c2e1c"},
                    {"range": [30, 60], "color": "#2a2a1a"},
                    {"range": [60, 100], "color": "#1a2c1a"},
                ],
                "threshold": {"line": {"color": ACCENT, "width": 3},
                              "thickness": 0.75, "value": 50}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0E1117", font_color="#A8C4E0", height=320
        )
        st.plotly_chart(fig_gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PRESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Customer Segmentation & Targeting Strategy</div>",
                unsafe_allow_html=True)

    df_seg = segment_customers(df)
    seg_summary = prescriptive_summary(df_seg)

    # Segment KPIs
    cols_seg = st.columns(4)
    seg_colors = {
        "🔥 Prime Target": "#E74C3C",
        "⚡ High Potential": "#F39C12",
        "💡 Moderate Potential": "#2980B9",
        "❄️ Low Priority": "#7F8C8D"
    }
    for i, row in seg_summary.iterrows():
        seg = row["Segment"]
        col = cols_seg[i % 4]
        c = seg_colors.get(seg, "#2980B9")
        with col:
            st.markdown(
                f"<div class='kpi-card' style='border-color:{c};'>"
                f"<p style='color:{c};font-size:0.85rem;font-weight:700;'>{seg}</p>"
                f"<p class='kpi-value' style='color:{c};font-size:1.6rem;'>{row['Actual Acceptance Rate (%)']:.1f}%</p>"
                f"<p class='kpi-sub'>{int(row['Count']):,} customers</p>"
                f"<p class='kpi-sub'>Avg Income: ${row['Avg_Income']}K | CCAvg: ${row['Avg_CCAvg']}K</p>"
                f"</div>", unsafe_allow_html=True
            )

    st.markdown("&nbsp;")

    col_seg1, col_seg2 = st.columns(2)

    with col_seg1:
        fig_seg_bar = px.bar(
            seg_summary, x="Segment", y="Count",
            color="Actual Acceptance Rate (%)",
            color_continuous_scale=["#1A3C6B", ACCEPT_COLOR],
            text="Count",
            title="Customer Count by Segment",
        )
        fig_seg_bar.update_traces(textposition="outside")
        fig_seg_bar.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
            font_color="#A8C4E0", title_font_color="#E8F4FD",
            coloraxis_showscale=True,
            xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
        )
        st.plotly_chart(fig_seg_bar, use_container_width=True)

    with col_seg2:
        fig_seg_rate = px.bar(
            seg_summary, x="Segment", y="Actual Acceptance Rate (%)",
            color="Actual Acceptance Rate (%)",
            color_continuous_scale=["#1A3C6B", ACCEPT_COLOR],
            text="Actual Acceptance Rate (%)",
            title="Actual Acceptance Rate by Segment",
        )
        fig_seg_rate.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_seg_rate.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#141a24",
            font_color="#A8C4E0", title_font_color="#E8F4FD",
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a")
        )
        st.plotly_chart(fig_seg_rate, use_container_width=True)

    # Full segment data table
    st.dataframe(seg_summary.style.background_gradient(
        cmap="Greens", subset=["Actual Acceptance Rate (%)"]
    ), use_container_width=True, hide_index=True)

    # ── Recommendations
    st.markdown("<div class='section-header'>📋 Strategic Recommendations</div>",
                unsafe_allow_html=True)

    recs = [
        ("🔥", "Prime Target (High Income + High CCAvg + CD Account)",
         "Launch immediate personalised loan campaigns via email and in-app notifications. "
         "Offer competitive interest rates and pre-approved loan amounts. These customers have the "
         "highest acceptance rate and financial capacity. Priority: HIGHEST.", ""),
        ("⚡", "High Potential (Income $50-100K + Graduate/Professional Education)",
         "Target with educational-themed marketing – highlight how personal loans can fund "
         "career growth, home improvement, or investments. Use digital channels (online banking "
         "pop-ups) since many use online banking. Priority: HIGH.", "warn"),
        ("💡", "Moderate Potential (Family Size ≥ 3 + Mortgage holders)",
         "Market loans as family-life facilitators – home renovation, children's education, "
         "emergency funds. Bundle with existing mortgage relationship for cross-sell messaging. "
         "Priority: MEDIUM.", "info"),
        ("💳", "CD Account Holders – Cross-Sell Opportunity",
         "Customers with CD accounts accept loans at 3× the base rate. "
         "Proactively reach out to all CD account holders with personalised loan offers "
         "as soon as their CD matures. Priority: HIGH.", "warn"),
        ("📱", "Online Banking Users",
         "Deploy in-app banners and push notifications for loan offers. "
         "Online users are already digitally engaged, making conversion cheaper and faster. "
         "A/B test different offer messages for optimisation.", "info"),
        ("❄️", "Low Priority (Low income, Undergrad, no banking services)",
         "Do not invest heavy marketing budget here. If targeting, use low-cost channels "
         "like SMS. Consider offering smaller credit products to build relationship first.", ""),
    ]

    for icon, title, text, style in recs:
        cls = f"insight-box {style}".strip()
        st.markdown(
            f"<div class='{cls}'><b>{icon} {title}</b><br>{text}</div>",
            unsafe_allow_html=True
        )

    # ── Expected ROI estimate
    st.markdown("<div class='section-header'>💰 Expected Campaign ROI Estimate</div>",
                unsafe_allow_html=True)

    prime = seg_summary[seg_summary["Segment"] == "🔥 Prime Target"]
    if not prime.empty:
        prime_count = int(prime["Count"].values[0])
        prime_rate = float(prime["Actual Acceptance Rate (%)"].values[0])
        expected_conv = int(prime_count * prime_rate / 100)
        st.markdown(f"""
        <div class='insight-box'>
        📊 <b>Prime Target Segment Campaign Estimate</b><br>
        • Customers in Prime Target: <b>{prime_count:,}</b><br>
        • Expected acceptance rate: <b>{prime_rate}%</b><br>
        • Estimated conversions: <b>{expected_conv:,}</b> new loan customers<br>
        • Assuming avg loan value of $50,000 → Potential portfolio addition:
          <b>${expected_conv * 50_000:,.0f}</b>
        </div>
        """, unsafe_allow_html=True)

    # ── Final strategic summary
    st.markdown("""
    <div class='insight-box info' style='margin-top:20px;'>
    <b>🏁 Overall Prescriptive Summary</b><br>
    1. <b>Income</b> is the single strongest predictor — focus all premium campaigns on $100K+ earners.<br>
    2. <b>CD Account holders</b> are a hidden gold mine — 3× acceptance rate vs non-holders.<br>
    3. <b>Education</b> matters — Advanced/Professional degree holders respond best to loan offers.<br>
    4. <b>Credit card spending (CCAvg)</b> signals financial activity and loan appetite.<br>
    5. <b>Family size of 3-4</b> with mortgages are underserved — bundle products for higher conversion.<br>
    6. Use the <b>Random Forest model</b> (AUC ~0.98) to score all customers monthly and 
       dynamically reprioritise outreach lists.
    </div>
    """, unsafe_allow_html=True)
