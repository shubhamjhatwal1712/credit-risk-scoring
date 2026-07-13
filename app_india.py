import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "india_credit_risk.csv"
MODEL_DIR = BASE_DIR / "models_india"
METRICS_PATH = MODEL_DIR / "metrics.json"

st.set_page_config(page_title="India Credit Risk Scoring", layout="wide", page_icon="₹")

theme_choice = st.sidebar.radio("Theme", ["Light", "Black"], horizontal=True)
BLACK_THEME = theme_choice == "Black"


def apply_theme():
    if not BLACK_THEME:
        return
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stHeader"], [data-testid="stSidebar"] {
            background: #05070a;
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] *, h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #e5e7eb;
        }
        [data-testid="stMetric"], [data-testid="stForm"], [data-testid="stDataFrame"] {
            background: #0b1117;
            border: 1px solid #1f2937;
            border-radius: 8px;
        }
        .stButton > button {
            background: #2563eb;
            color: #ffffff;
            border-color: #3b82f6;
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


apply_theme()

CATEGORICAL_COLUMNS = [
    "state_region",
    "city_tier",
    "employment_type",
    "loan_purpose",
    "residential_status",
    "collateral_type",
]


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
def load_data():
    return pd.read_csv(DATA_PATH)


model, scaler, encoders, feature_names, explainer, metrics = load_artifacts()
df = load_data()

st.title("India Credit Risk Scoring System")
st.caption("Demo lending-risk model using an India-style synthetic dataset. Replace with real lender data when available.")

tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "Score an Applicant", "Model Performance"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{len(df):,}")
    c2.metric("Default Rate", f"{df['defaulted'].mean():.1%}")
    c3.metric("Avg Loan Amount", f"₹{df['loan_amount_inr'].mean():,.0f}")
    c4.metric("Avg Credit Score", f"{df['credit_score'].mean():.0f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Default Rate by Employment Type")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        rate = df.groupby("employment_type")["defaulted"].mean().sort_values() * 100
        rate.plot(kind="barh", ax=ax, color="#c0392b")
        ax.set_xlabel("Default Rate (%)")
        style_chart(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Default Rate by City Tier")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        rate = df.groupby("city_tier")["defaulted"].mean().sort_values() * 100
        rate.plot(kind="barh", ax=ax, color="#2980b9")
        ax.set_xlabel("Default Rate (%)")
        style_chart(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Default Rate by Loan Purpose")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        rate = df.groupby("loan_purpose")["defaulted"].mean().sort_values() * 100
        rate.plot(kind="barh", ax=ax, color="#8e44ad")
        ax.set_xlabel("Default Rate (%)")
        style_chart(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        st.subheader("Top Global Risk Drivers")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        pd.Series(metrics["global_shap_importance"]).sort_values().plot(kind="barh", ax=ax, color="#16a085")
        ax.set_xlabel("Mean |SHAP value|")
        style_chart(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    st.subheader("Score an Indian loan applicant")
    with st.form("india_credit_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            state_region = st.selectbox("State Region", list(encoders["state_region"].classes_))
            city_tier = st.selectbox("City Tier", list(encoders["city_tier"].classes_))
            employment_type = st.selectbox("Employment Type", list(encoders["employment_type"].classes_))
            monthly_income_inr = st.slider("Monthly Income (INR)", 12000, 350000, 55000, step=1000)
            age = st.slider("Age", 21, 65, 34)
            loan_amount_inr = st.slider("Loan Amount (INR)", 20000, 2500000, 350000, step=10000)
            loan_tenure_months = st.selectbox("Loan Tenure (months)", [6, 12, 18, 24, 36, 48, 60, 84], index=4)
        with col2:
            loan_purpose = st.selectbox("Loan Purpose", list(encoders["loan_purpose"].classes_))
            existing_emis_inr = st.slider("Existing EMIs (INR)", 0, 160000, 12000, step=1000)
            credit_score = st.slider("Credit Score", 300, 900, 700)
            prior_defaults = st.selectbox("Prior Defaults", [0, 1])
            active_loans = st.slider("Active Loans", 0, 7, 1)
            bank_account_age_years = st.slider("Bank Account Age (years)", 0.2, 25.0, 5.0, step=0.1)
        with col3:
            digital_payment_score = st.slider("Digital Payment Score", 5, 100, 70)
            residential_status = st.selectbox("Residential Status", list(encoders["residential_status"].classes_))
            collateral_type = st.selectbox("Collateral Type", list(encoders["collateral_type"].classes_))
            dependents = st.slider("Dependents", 0, 4, 2)
            delinquency_30dpd_last_12m = st.slider("30+ DPD Events Last 12 Months", 0, 6, 0)

        submitted = st.form_submit_button("Score Applicant", type="primary")

    if submitted:
        estimated_emi_inr = loan_amount_inr / loan_tenure_months * 0.095
        debt_to_income_pct = min(((existing_emis_inr + estimated_emi_inr) / monthly_income_inr) * 100, 180)
        loan_to_income_ratio = min(loan_amount_inr / (monthly_income_inr * 12), 8)

        input_df = pd.DataFrame(
            [
                {
                    "state_region": state_region,
                    "city_tier": city_tier,
                    "employment_type": employment_type,
                    "monthly_income_inr": monthly_income_inr,
                    "age": age,
                    "loan_amount_inr": loan_amount_inr,
                    "loan_tenure_months": loan_tenure_months,
                    "loan_purpose": loan_purpose,
                    "existing_emis_inr": existing_emis_inr,
                    "credit_score": credit_score,
                    "prior_defaults": prior_defaults,
                    "active_loans": active_loans,
                    "bank_account_age_years": bank_account_age_years,
                    "digital_payment_score": digital_payment_score,
                    "residential_status": residential_status,
                    "collateral_type": collateral_type,
                    "dependents": dependents,
                    "delinquency_30dpd_last_12m": delinquency_30dpd_last_12m,
                    "debt_to_income_pct": round(debt_to_income_pct, 1),
                    "loan_to_income_ratio": round(loan_to_income_ratio, 2),
                }
            ]
        )

        for col, encoder in encoders.items():
            input_df[col] = encoder.transform(input_df[col])
        input_df = input_df[feature_names]
        X_for_pred = scaler.transform(input_df) if model.__class__.__name__ == "LogisticRegression" else input_df
        prob_default = model.predict_proba(X_for_pred)[0][1]
        st.session_state["india_last_score"] = {"prob_default": float(prob_default), "X_for_pred": X_for_pred}

    if "india_last_score" in st.session_state:
        prob_default = st.session_state["india_last_score"]["prob_default"]
        X_for_pred = st.session_state["india_last_score"]["X_for_pred"]

        st.divider()
        risk_col, gauge_col = st.columns([1, 2])
        with risk_col:
            if prob_default >= 0.5:
                st.error(f"### High Risk\n**{prob_default:.1%}** predicted default probability")
            elif prob_default >= 0.3:
                st.warning(f"### Manual Review\n**{prob_default:.1%}** predicted default probability")
            else:
                st.success(f"### Low Risk\n**{prob_default:.1%}** predicted default probability")
        with gauge_col:
            st.progress(min(int(prob_default * 100), 100))
            st.caption(
                "Recommended action: "
                + (
                    "Decline or request collateral/guarantor." if prob_default >= 0.5 else
                    "Manual review recommended." if prob_default >= 0.3 else
                    "Approve subject to normal checks."
                )
            )

        st.subheader("SHAP Explanation")
        if st.button("Generate SHAP Explanation"):
            with st.spinner("Generating explanation..."):
                try:
                    shap_vals = explainer.shap_values(X_for_pred)
                    shap_vals = np.asarray(shap_vals)
                    if shap_vals.ndim == 3:
                        shap_vals = shap_vals[:, :, 1]
                    contrib = pd.Series(shap_vals[0], index=feature_names)
                    top_contrib = contrib.reindex(contrib.abs().sort_values().tail(10).index).sort_values()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    top_contrib.plot(
                        kind="barh",
                        ax=ax,
                        color=["#c0392b" if value > 0 else "#2980b9" for value in top_contrib.values],
                    )
                    ax.set_xlabel("SHAP value (positive pushes toward default)")
                    style_chart(fig, ax)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as exc:
                    st.info(f"SHAP explanation unavailable for this input: {exc}")

with tab3:
    st.subheader(f"Best Model: {metrics['best_model']}")
    comp_df = pd.DataFrame(metrics["all_results"]).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    highlight_color = "#14532d" if BLACK_THEME else "#d5f5e3"
    st.dataframe(comp_df.style.highlight_max(axis=0, color=highlight_color), use_container_width=True)
    st.caption("This India dataset is synthetic and intended for portfolio/demo use, not production lending decisions.")
