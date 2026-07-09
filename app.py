"""
Loan Default / Credit Risk Dashboard
=======================================
Interactive Streamlit app for:
  1. Portfolio-level risk overview (for a lending/credit team)
  2. Live credit risk scoring for a single applicant, with SHAP explanation
     (which factors pushed the decision, and by how much)

Run locally:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "credit_risk.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"

st.set_page_config(page_title="Credit Risk Scoring", layout="wide", page_icon="🏦")

theme_choice = st.sidebar.radio("Theme", ["Light", "Black"], horizontal=True)
BLACK_THEME = theme_choice == "Black"


def apply_theme(is_black: bool):
    if not is_black:
        return

    st.markdown(
        """
        <style>
        .stApp {
            background: #05070a;
            color: #e5e7eb;
        }
        [data-testid="stHeader"],
        [data-testid="stSidebar"] {
            background: #05070a;
        }
        [data-testid="stSidebar"] * {
            color: #e5e7eb;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #e5e7eb;
        }
        [data-testid="stMetric"],
        [data-testid="stForm"],
        [data-testid="stDataFrame"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #0b1117;
            border: 1px solid #1f2937;
            border-radius: 8px;
        }
        .stButton > button {
            background: #2563eb;
            color: #ffffff;
            border-color: #3b82f6;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            background: #0b1117;
            border-radius: 8px 8px 0 0;
            border: 1px solid #1f2937;
        }
        .stTabs [aria-selected="true"] {
            background: #111827;
            border-bottom-color: #2563eb;
        }
        hr {
            border-color: #1f2937;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, ax):
    if not BLACK_THEME:
        return

    fig.patch.set_facecolor("#05070a")
    ax.set_facecolor("#0b1117")
    ax.tick_params(colors="#d1d5db")
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")
    ax.title.set_color("#e5e7eb")
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.grid(color="#1f2937", alpha=0.45)


apply_theme(BLACK_THEME)

COLUMN_NAMES = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_duration", "installment_rate",
    "personal_status_sex", "other_debtors", "present_residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "credit_risk"
]

CODE_MAPS = {
    "checking_account_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no checking account"},
    "credit_history": {"A30": "no credits taken", "A31": "all paid duly (this bank)", "A32": "existing credits paid duly", "A33": "delay in past", "A34": "critical/other credits existing"},
    "purpose": {"A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment", "A43": "radio/TV", "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business", "A410": "other"},
    "savings_account": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM", "A65": "unknown/none"},
    "employment_duration": {"A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years", "A74": "4-7 years", "A75": ">= 7 years"},
    "personal_status_sex": {"A91": "male: divorced/separated", "A92": "female: divorced/separated/married", "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings agreement/life insurance", "A123": "car or other", "A124": "unknown/none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemployed/unskilled non-resident", "A172": "unskilled resident", "A173": "skilled employee", "A174": "management/self-employed/highly qualified"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "credit_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return model, scaler, encoders, feature_names, explainer, metrics


@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
    for col, mapping in CODE_MAPS.items():
        df[col] = df[col].map(mapping)
    df["credit_risk_label"] = df["credit_risk"].map({1: "Good", 2: "Bad"})
    return df


model, scaler, encoders, feature_names, explainer, metrics = load_artifacts()
raw_df = load_raw_data()

st.title("🏦 Loan Default / Credit Risk Scoring System")
st.caption("Predicting applicant credit risk with explainable AI · Statlog German Credit dataset (1,000 applicants)")

tab1, tab2, tab3 = st.tabs(["📈 Portfolio Overview", "🔮 Score an Applicant", "🧠 Model Performance"])

# ---------------------------------------------------------------------------
# TAB 1: PORTFOLIO OVERVIEW
# ---------------------------------------------------------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    bad_rate = (raw_df["credit_risk"] == 2).mean()
    col1.metric("Total Applicants", f"{len(raw_df):,}")
    col2.metric("Bad Risk Rate", f"{bad_rate:.1%}")
    col3.metric("Avg Credit Amount", f"{raw_df['credit_amount'].mean():,.0f} DM")
    col4.metric("Avg Loan Duration", f"{raw_df['duration_months'].mean():.0f} months")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Rate by Checking Account Status")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ct = pd.crosstab(raw_df["checking_account_status"], raw_df["credit_risk_label"], normalize="index") * 100
        ct["Bad"].sort_values().plot(kind="barh", ax=ax, color="#c0392b")
        ax.set_xlabel("Bad Risk Rate (%)")
        style_chart(fig, ax)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Applicants with no checking account history default far less than those with a visibly overdrawn account.")

    with c2:
        st.subheader("Risk Rate by Credit History")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ct2 = pd.crosstab(raw_df["credit_history"], raw_df["credit_risk_label"], normalize="index") * 100
        ct2["Bad"].sort_values().plot(kind="barh", ax=ax, color="#2980b9")
        ax.set_xlabel("Bad Risk Rate (%)")
        style_chart(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Risk Rate by Loan Purpose")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ct3 = pd.crosstab(raw_df["purpose"], raw_df["credit_risk_label"], normalize="index") * 100
        ct3["Bad"].sort_values().plot(kind="barh", ax=ax, color="#8e44ad")
        ax.set_xlabel("Bad Risk Rate (%)")
        style_chart(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

    with c4:
        st.subheader("Top Global Risk Drivers (SHAP)")
        if metrics.get("global_shap_importance"):
            fi = pd.Series(metrics["global_shap_importance"]).sort_values()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fi.plot(kind="barh", ax=ax, color="#16a085")
            ax.set_xlabel("Mean |SHAP value|")
            style_chart(fig, ax)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("These are the features that move the model's decision most, averaged across all applicants.")

# ---------------------------------------------------------------------------
# TAB 2: SCORE AN APPLICANT
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Score a loan applicant")
    st.write("Enter applicant details to get a credit risk decision, with a SHAP breakdown showing exactly why the model decided what it did — the transparency a real lending decision requires.")

    with st.form("credit_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            checking = st.selectbox("Checking Account Status", list(CODE_MAPS["checking_account_status"].values()))
            duration = st.slider("Loan Duration (months)", 4, 72, 24)
            history = st.selectbox("Credit History", list(CODE_MAPS["credit_history"].values()))
            purpose = st.selectbox("Loan Purpose", list(CODE_MAPS["purpose"].values()))
            amount = st.slider("Credit Amount (DM)", 250, 20000, 3000)
            savings = st.selectbox("Savings Account", list(CODE_MAPS["savings_account"].values()))
            employment = st.selectbox("Employment Duration", list(CODE_MAPS["employment_duration"].values()))

        with col2:
            installment_rate = st.slider("Installment Rate (% of income)", 1, 4, 2)
            personal_status = st.selectbox("Personal Status", list(CODE_MAPS["personal_status_sex"].values()))
            other_debtors = st.selectbox("Other Debtors/Guarantors", list(CODE_MAPS["other_debtors"].values()))
            residence_since = st.slider("Years at Present Residence", 1, 4, 2)
            property_val = st.selectbox("Property", list(CODE_MAPS["property"].values()))
            age = st.slider("Age", 18, 75, 35)

        with col3:
            other_installments = st.selectbox("Other Installment Plans", list(CODE_MAPS["other_installment_plans"].values()))
            housing = st.selectbox("Housing", list(CODE_MAPS["housing"].values()))
            existing_credits = st.slider("Existing Credits at This Bank", 1, 4, 1)
            job = st.selectbox("Job", list(CODE_MAPS["job"].values()))
            num_dependents = st.slider("Number of Dependents", 1, 2, 1)
            telephone = st.selectbox("Telephone", list(CODE_MAPS["telephone"].values()))
            foreign_worker = st.selectbox("Foreign Worker", list(CODE_MAPS["foreign_worker"].values()))

        submitted = st.form_submit_button("Score Applicant", type="primary")

    if submitted:
        input_dict = {
            "checking_account_status": checking, "duration_months": duration,
            "credit_history": history, "purpose": purpose, "credit_amount": amount,
            "savings_account": savings, "employment_duration": employment,
            "installment_rate": installment_rate, "personal_status_sex": personal_status,
            "other_debtors": other_debtors, "present_residence_since": residence_since,
            "property": property_val, "age": age, "other_installment_plans": other_installments,
            "housing": housing, "existing_credits": existing_credits, "job": job,
            "num_dependents": num_dependents, "telephone": telephone,
            "foreign_worker": foreign_worker,
        }
        try:
            df_input = pd.DataFrame([input_dict])

            # Feature engineering (must mirror train_model.py exactly)
            df_input["debt_to_duration"] = df_input["credit_amount"] / df_input["duration_months"]
            df_input["installment_burden"] = df_input["installment_rate"] * df_input["credit_amount"] / 100
            df_input["age_group_young"] = (df_input["age"] < 30).astype(int)
            df_input["age_group_senior"] = (df_input["age"] >= 60).astype(int)
            df_input["has_no_checking_account"] = (df_input["checking_account_status"] == "no checking account").astype(int)
            df_input["long_term_loan"] = (df_input["duration_months"] > 24).astype(int)
            df_input["high_credit_amount"] = (df_input["credit_amount"] > raw_df["credit_amount"].median()).astype(int)

            for col, le in encoders.items():
                if col in df_input.columns:
                    df_input[col] = df_input[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df_input[col] = le.transform(df_input[col])

            df_input = df_input[feature_names]

            if model.__class__.__name__ == "LogisticRegression":
                X_for_pred = scaler.transform(df_input)
            else:
                X_for_pred = df_input

            prob_bad = model.predict_proba(X_for_pred)[0][1]
            st.session_state["last_score"] = {
                "prob_bad": float(prob_bad),
                "X_for_pred": X_for_pred,
            }
        except Exception as e:
            st.error(f"Unable to score this applicant: {e}")

    if "last_score" in st.session_state:
        prob_bad = st.session_state["last_score"]["prob_bad"]
        X_for_pred = st.session_state["last_score"]["X_for_pred"]

        st.divider()
        risk_col, gauge_col = st.columns([1, 2])
        with risk_col:
            if prob_bad >= 0.5:
                st.error(f"### ⚠️ Bad Risk\n**{prob_bad:.1%}** predicted default probability")
            elif prob_bad >= 0.3:
                st.warning(f"### Borderline\n**{prob_bad:.1%}** predicted default probability")
            else:
                st.success(f"### ✅ Good Risk\n**{prob_bad:.1%}** predicted default probability")

        with gauge_col:
            st.progress(min(int(prob_bad * 100), 100))
            st.caption("Recommended action: " + (
                "Decline or require additional collateral/guarantor." if prob_bad >= 0.5 else
                "Manual review recommended before approval." if prob_bad >= 0.3 else
                "Approve — low predicted default risk."
            ))

        st.divider()
        st.subheader("Why the model made this decision (SHAP)")
        st.caption("The risk score is ready. Generate the explanation when you want the detailed factor breakdown.")

        if st.button("Generate SHAP Explanation"):
            with st.spinner("Generating explanation..."):
                try:
                    shap_vals = explainer.shap_values(X_for_pred)
                    shap_vals = np.asarray(shap_vals)
                    if shap_vals.ndim == 3:
                        shap_vals = shap_vals[:, :, 1]
                    row_shap = shap_vals[0]

                    contrib = pd.Series(row_shap, index=feature_names)
                    top_contrib = contrib.reindex(contrib.abs().sort_values().tail(10).index).sort_values()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    top_contrib.plot(kind="barh", ax=ax, color=[
                        "#c0392b" if v > 0 else "#2980b9" for v in top_contrib.values
                    ])
                    ax.set_xlabel("SHAP value (positive pushes toward Bad Risk, negative toward Good Risk)")
                    style_chart(fig, ax)
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption("Red bars increased predicted default risk; blue bars decreased it.")
                except Exception as e:
                    st.info(f"SHAP explanation unavailable for this input: {e}")

# ---------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE
# ---------------------------------------------------------------------------
with tab3:
    st.subheader(f"Best Model: {metrics['best_model']}")
    results = metrics["all_results"]
    comp_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    highlight_color = "#14532d" if BLACK_THEME else "#d5f5e3"
    st.dataframe(comp_df.style.highlight_max(axis=0, color=highlight_color), use_container_width=True)
    st.caption(
        "Recall on 'Bad Risk' is weighted heavily — in lending, approving a bad-risk applicant "
        "costs far more than declining a good one (reflected in the original Statlog cost matrix: "
        "misclassifying bad-as-good is penalized 5x higher than good-as-bad)."
    )

st.divider()
st.caption("Built by Shubham · Python, Scikit-learn, XGBoost, SHAP, Streamlit · Statlog German Credit Data (UCI)")
