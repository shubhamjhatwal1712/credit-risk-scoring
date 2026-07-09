"""
India Credit Risk Prediction - Training Pipeline
================================================
Builds a demo India-focused credit risk model using a synthetic lending dataset.

The dataset is generated locally because real Indian credit bureau / lender data
is not public. Replace data/india_credit_risk.csv with a real CSV later if you
have one with the same columns.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models_india"
DATA_PATH = DATA_DIR / "india_credit_risk.csv"
RANDOM_STATE = 42

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLUMNS = [
    "state_region",
    "city_tier",
    "employment_type",
    "loan_purpose",
    "residential_status",
    "collateral_type",
]


def generate_india_credit_data(path: Path, n_rows: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    state_region = rng.choice(
        ["North", "South", "West", "East", "Central", "Northeast"],
        size=n_rows,
        p=[0.24, 0.24, 0.22, 0.14, 0.10, 0.06],
    )
    city_tier = rng.choice(["Metro", "Tier 1", "Tier 2", "Tier 3/Rural"], size=n_rows, p=[0.22, 0.26, 0.30, 0.22])
    employment_type = rng.choice(
        ["Salaried", "Self-employed", "Gig/Contract", "Business owner", "Unemployed"],
        size=n_rows,
        p=[0.46, 0.24, 0.13, 0.13, 0.04],
    )
    loan_purpose = rng.choice(
        ["Personal", "Two-wheeler", "Education", "Business", "Home improvement", "Medical", "Consumer durable"],
        size=n_rows,
        p=[0.26, 0.16, 0.12, 0.17, 0.10, 0.08, 0.11],
    )
    residential_status = rng.choice(["Owned", "Rented", "Family owned", "Company provided"], size=n_rows, p=[0.33, 0.37, 0.25, 0.05])
    collateral_type = rng.choice(["None", "Gold", "Vehicle", "Property", "Fixed deposit"], size=n_rows, p=[0.54, 0.18, 0.14, 0.09, 0.05])

    age = np.clip(rng.normal(36, 10, n_rows).round(), 21, 65).astype(int)
    monthly_income_inr = np.clip(rng.lognormal(mean=10.65, sigma=0.55, size=n_rows), 12000, 350000).round(-2).astype(int)
    loan_amount_inr = np.clip(rng.lognormal(mean=12.25, sigma=0.65, size=n_rows), 20000, 2500000).round(-3).astype(int)
    loan_tenure_months = rng.choice([6, 12, 18, 24, 36, 48, 60, 84], size=n_rows, p=[0.08, 0.17, 0.12, 0.22, 0.18, 0.11, 0.08, 0.04])
    existing_emis_inr = np.clip(monthly_income_inr * rng.beta(1.5, 5.5, n_rows), 0, 160000).round(-2).astype(int)
    active_loans = rng.poisson(1.1, n_rows).clip(0, 7)
    dependents = rng.choice([0, 1, 2, 3, 4], size=n_rows, p=[0.18, 0.29, 0.30, 0.16, 0.07])
    bank_account_age_years = np.clip(rng.gamma(3.0, 2.2, n_rows), 0.2, 25).round(1)
    digital_payment_score = np.clip(rng.normal(67, 18, n_rows), 5, 100).round().astype(int)
    prior_defaults = rng.binomial(1, 0.12, n_rows)
    delinquency_30dpd_last_12m = rng.poisson(0.35 + prior_defaults * 0.9, n_rows).clip(0, 6)

    base_credit = rng.normal(710, 75, n_rows)
    credit_score = (
        base_credit
        - prior_defaults * 95
        - delinquency_30dpd_last_12m * 28
        - (employment_type == "Unemployed") * 65
        - (employment_type == "Gig/Contract") * 25
        + (collateral_type != "None") * 25
        + (bank_account_age_years > 5) * 18
        + (digital_payment_score > 75) * 14
    )
    credit_score = np.clip(credit_score, 300, 900).round().astype(int)

    estimated_emi_inr = loan_amount_inr / loan_tenure_months * 0.095
    debt_to_income_pct = np.clip(((existing_emis_inr + estimated_emi_inr) / monthly_income_inr) * 100, 0, 180).round(1)
    loan_to_income_ratio = np.clip(loan_amount_inr / (monthly_income_inr * 12), 0.05, 8).round(2)

    risk_score = (
        -4.2
        + 0.030 * debt_to_income_pct
        + 0.45 * loan_to_income_ratio
        - 0.0065 * (credit_score - 650)
        + 1.25 * prior_defaults
        + 0.36 * delinquency_30dpd_last_12m
        + 0.20 * active_loans
        + 0.45 * (employment_type == "Unemployed")
        + 0.30 * (employment_type == "Gig/Contract")
        + 0.22 * (city_tier == "Tier 3/Rural")
        - 0.42 * (collateral_type != "None")
        - 0.20 * (residential_status == "Owned")
        - 0.012 * digital_payment_score
    )
    default_probability = 1 / (1 + np.exp(-risk_score))
    defaulted = rng.binomial(1, default_probability)

    df = pd.DataFrame(
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
            "debt_to_income_pct": debt_to_income_pct,
            "loan_to_income_ratio": loan_to_income_ratio,
            "defaulted": defaulted,
        }
    )
    df.to_csv(path, index=False)
    return df


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return generate_india_credit_data(DATA_PATH)


def encode_categoricals(df: pd.DataFrame):
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def evaluate_model(model, X_test, y_test, scaled=False, scaler=None):
    X_eval = scaler.transform(X_test) if scaled else X_test
    preds = model.predict(X_eval)
    probs = model.predict_proba(X_eval)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def main():
    print("Loading India credit dataset...")
    df = load_data()
    encoded_df, encoders = encode_categoricals(df)

    X = encoded_df.drop(columns=["defaulted"])
    y = encoded_df["defaulted"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
    }
    if XGBClassifier is not None:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.04,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        )
    else:
        print("XGBoost is not installed; skipping the XGBoost candidate.")

    fitted = {}
    results = {}
    for name, model in candidates.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            results[name] = evaluate_model(model, X_test, y_test, scaled=True, scaler=scaler)
        else:
            model.fit(X_train, y_train)
            results[name] = evaluate_model(model, X_test, y_test)
        fitted[name] = model
        print(name, results[name])

    best_name = max(results, key=lambda key: results[key]["roc_auc"])
    best_model = fitted[best_name]

    if best_name == "Logistic Regression":
        X_train_for_shap = scaler.transform(X_train)
        X_test_for_shap = scaler.transform(X_test)
        explainer = shap.LinearExplainer(best_model, X_train_for_shap)
    else:
        X_test_for_shap = X_test
        explainer = shap.TreeExplainer(best_model)

    shap_values = explainer.shap_values(X_test_for_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    global_importance = dict(
        sorted(
            zip(feature_names, np.abs(shap_values).mean(axis=0).flatten().tolist()),
            key=lambda item: -item[1],
        )[:10]
    )

    joblib.dump(best_model, MODEL_DIR / "credit_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoders, MODEL_DIR / "encoders.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(
            {
                "dataset": "Synthetic India Credit Risk",
                "best_model": best_name,
                "all_results": results,
                "global_shap_importance": global_importance,
                "dataset_size": len(df),
                "default_rate": round(float(y.mean()), 4),
            },
            f,
            indent=2,
        )

    print(f"Saved India dataset to {DATA_PATH}")
    print(f"Saved India model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
