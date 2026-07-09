"""
Loan Default / Credit Risk Prediction - Training Pipeline
=============================================================
Predicts whether a loan applicant is a good or bad credit risk.
Dataset: Statlog German Credit Data (UCI, Hofmann 1994) - 1,000 applicants, 20 attributes.

Includes SHAP explainability so every prediction can be justified feature-by-feature -
critical in lending, where decisions have to be explainable to applicants and regulators.

Author: Shubham
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import shap

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

MODEL_DIR = Path(__file__).resolve().parent
BASE_DIR = MODEL_DIR.parent
DATA_PATH = BASE_DIR / "data" / "credit_risk.csv"
OUTPUT_DIR = MODEL_DIR
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

COLUMN_NAMES = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_duration", "installment_rate",
    "personal_status_sex", "other_debtors", "present_residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "credit_risk"
]

# Human-readable decodes for the Statlog attribute codes (makes the dashboard
# and SHAP explanations legible instead of "A93 increases risk by 0.14")
CODE_MAPS = {
    "checking_account_status": {
        "A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no checking account"},
    "credit_history": {
        "A30": "no credits taken", "A31": "all paid duly (this bank)",
        "A32": "existing credits paid duly", "A33": "delay in past",
        "A34": "critical/other credits existing"},
    "purpose": {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
        "A43": "radio/TV", "A44": "domestic appliances", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining",
        "A49": "business", "A410": "other"},
    "savings_account": {
        "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
        "A64": ">= 1000 DM", "A65": "unknown/none"},
    "employment_duration": {
        "A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years",
        "A74": "4-7 years", "A75": ">= 7 years"},
    "personal_status_sex": {
        "A91": "male: divorced/separated", "A92": "female: divorced/separated/married",
        "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {
        "A121": "real estate", "A122": "savings agreement/life insurance",
        "A123": "car or other", "A124": "unknown/none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed/unskilled non-resident", "A172": "unskilled resident",
        "A173": "skilled employee", "A174": "management/self-employed/highly qualified"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN + DECODE
# ---------------------------------------------------------------------------
def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)

    for col, mapping in CODE_MAPS.items():
        df[col] = df[col].map(mapping)

    # Original target: 1 = Good, 2 = Bad -> convert to 0/1 where 1 = Bad (default risk)
    df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["debt_to_duration"] = df["credit_amount"] / df["duration_months"]
    df["installment_burden"] = df["installment_rate"] * df["credit_amount"] / 100
    df["age_group_young"] = (df["age"] < 30).astype(int)
    df["age_group_senior"] = (df["age"] >= 60).astype(int)
    df["has_no_checking_account"] = (df["checking_account_status"] == "no checking account").astype(int)
    df["long_term_loan"] = (df["duration_months"] > 24).astype(int)
    df["high_credit_amount"] = (df["credit_amount"] > df["credit_amount"].median()).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame):
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


# ---------------------------------------------------------------------------
# 3. TRAIN + EVALUATE
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"),
    }

    if XGBClassifier is not None:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE, eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
        )
    else:
        print("XGBoost is not installed; skipping the XGBoost candidate.")

    results = {}
    fitted_models = {}

    for name, model in candidates.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        fitted_models[name] = model
        print(f"\n{name}")
        print(classification_report(y_test, preds, target_names=["Good Risk", "Bad Risk"]))

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best_name} ({results[best_name]['roc_auc']})")

    return fitted_models, results, best_name, scaler


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Loading, cleaning, decoding data...")
    df = load_and_clean(DATA_PATH)

    print("Engineering features...")
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)

    X = df.drop(columns=["credit_risk"])
    y = df["credit_risk"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Training models...")
    fitted_models, results, best_name, scaler = train_and_evaluate(X_train, X_test, y_train, y_test)
    best_model = fitted_models[best_name]

    # --- SHAP explainability on the best model ---
    print("Computing SHAP values...")
    if best_name == "Logistic Regression":
        X_test_for_shap = scaler.transform(X_test)
        explainer = shap.LinearExplainer(best_model, scaler.transform(X_train))
    elif best_name == "Random Forest":
        X_test_for_shap = X_test
        explainer = shap.TreeExplainer(best_model)
    else:
        X_test_for_shap = X_test
        explainer = shap.TreeExplainer(best_model)

    shap_values = explainer.shap_values(X_test_for_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        # shape (n_samples, n_features, n_classes) -> take positive class
        shap_values = shap_values[:, :, 1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_abs_shap = np.asarray(mean_abs_shap).flatten()
    global_importance = dict(
        sorted(zip(feature_names, mean_abs_shap.tolist()), key=lambda x: -x[1])[:10]
    )

    # --- Save artifacts ---
    joblib.dump(best_model, MODEL_DIR / "credit_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoders, MODEL_DIR / "encoders.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "all_results": results,
            "global_shap_importance": global_importance,
            "dataset_size": len(df),
            "bad_risk_rate": round(y.mean(), 4),
        }, f, indent=2)

    print(f"\nArtifacts saved to {MODEL_DIR} and {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
