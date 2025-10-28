from pathlib import Path

import cloudpickle
import pandas as pd

from src.churn_features import MODEL_FEATURES, RAW_FEATURES, build_preprocessing_pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "smote_lgbm.pkl"
PIPELINE_PATH = BASE_DIR / "preprocessing_pipeline.pkl"

RISK_BANDS = (
    (0.50, "High", "At-risk customer. Prioritize a retention offer and support follow-up."),
    (0.20, "Medium", "Watch closely. Improve engagement with targeted content and check-ins."),
    (0.0, "Low", "Healthy customer. Maintain experience quality and monitor normally."),
)

MODEL_METRICS = {
    "Accuracy": "0.90",
    "ROC-AUC": "0.9579",
    "Model": "LightGBM + SMOTE",
    "Runtime": "Python 3.11",
}


def load_artifacts(model_path=MODEL_PATH, pipeline_path=PIPELINE_PATH):
    with Path(model_path).open("rb") as model_file:
        model = cloudpickle.load(model_file)

    pipeline_source = "saved artifact"
    try:
        with Path(pipeline_path).open("rb") as pipeline_file:
            pipeline = cloudpickle.load(pipeline_file)
    except (AttributeError, EOFError, ImportError, ModuleNotFoundError, TypeError):
        pipeline = build_preprocessing_pipeline()
        pipeline_source = "rebuilt from source"

    return model, pipeline, pipeline_source


def validate_customer(customer):
    warnings = []
    expected_total = customer["AccountAge"] * customer["MonthlyCharges"]

    if expected_total > 0:
        ratio = customer["TotalCharges"] / expected_total
        if ratio < 0.5 or ratio > 1.75:
            warnings.append(
                "Total Charges is unusual compared with Account Age and Monthly Charges."
            )

    if customer["SupportTicketsPerMonth"] >= 5 and customer["UserRating"] >= 4.5:
        warnings.append("High support tickets with a high rating may indicate inconsistent inputs.")

    if customer["ViewingHoursPerWeek"] == 0 and customer["AverageViewingDuration"] > 0:
        warnings.append("Average Viewing Duration is above zero while weekly viewing hours are zero.")

    return warnings


def risk_details(probability):
    for threshold, label, recommendation in RISK_BANDS:
        if probability >= threshold:
            return label, recommendation
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


def score_customers(model, pipeline, customers):
    missing_columns = [column for column in RAW_FEATURES if column not in customers.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")

    processed = pipeline.transform(customers[RAW_FEATURES])
    predictions = model.predict(processed)
    probabilities = model.predict_proba(processed)[:, 1]

    scored = customers.copy()
    scored["ChurnPrediction"] = predictions.astype(int)
    scored["ChurnProbability"] = probabilities.astype(float)
    scored["RiskBand"] = [risk_details(probability)[0] for probability in probabilities]
    scored["RecommendedAction"] = [risk_details(probability)[1] for probability in probabilities]
    return scored


def score_customer(model, pipeline, customer):
    scored = score_customers(model, pipeline, pd.DataFrame([customer]))
    row = scored.iloc[0]
    return {
        "prediction": int(row["ChurnPrediction"]),
        "probability": float(row["ChurnProbability"]),
        "risk_band": row["RiskBand"],
        "recommendation": row["RecommendedAction"],
        "warnings": validate_customer(customer),
    }


def model_driver_frame(model, limit=10):
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(MODEL_FEATURES):
        return pd.DataFrame(columns=["Feature", "Importance"])

    return (
        pd.DataFrame({"Feature": MODEL_FEATURES, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def explain_customer(customer):
    reasons = []

    if customer["SupportTicketsPerMonth"] >= 4:
        reasons.append("Support load is elevated, which can signal unresolved service friction.")
    if customer["UserRating"] <= 2.5:
        reasons.append("User rating is low, suggesting dissatisfaction with the experience.")
    if customer["ViewingHoursPerWeek"] <= 3:
        reasons.append("Weekly viewing engagement is low.")
    if customer["ContentDownloadsPerMonth"] <= 1:
        reasons.append("Content download activity is limited.")
    if customer["MonthlyCharges"] >= 90:
        reasons.append("Monthly charges are high, increasing price sensitivity.")
    if customer["WatchlistSize"] <= 2:
        reasons.append("Watchlist size is small, suggesting weak content intent.")
    if not reasons:
        reasons.append("No single behavioral warning dominates; risk is mainly model-driven.")

    return reasons


def retention_plan(risk_band):
    plans = {
        "High": [
            "Route to retention specialist within 24 hours.",
            "Offer plan optimization, discount, or pause option.",
            "Prioritize support follow-up for unresolved tickets.",
        ],
        "Medium": [
            "Trigger targeted content recommendations.",
            "Send engagement campaign within the next billing cycle.",
            "Monitor support and viewing behavior weekly.",
        ],
        "Low": [
            "Keep customer in normal monitoring.",
            "Use loyalty or upgrade messaging only when relevant.",
            "Avoid unnecessary discounting.",
        ],
    }
    return plans.get(risk_band, plans["Low"])
