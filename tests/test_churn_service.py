import numpy as np
import pandas as pd

from src.churn_features import RAW_FEATURES, build_preprocessing_pipeline
from src.churn_service import (
    explain_customer,
    retention_plan,
    score_customer,
    score_customers,
    validate_customer,
)


class DummyModel:
    feature_importances_ = np.arange(15)

    def predict(self, X):
        return np.array([1 if value >= 0.5 else 0 for value in self.predict_proba(X)[:, 1]])

    def predict_proba(self, X):
        probabilities = np.array([0.72 for _ in range(len(X))])
        return np.column_stack([1 - probabilities, probabilities])


def customer_row():
    return {
        "AccountAge": 12,
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
        "SubscriptionType": "Basic",
        "ViewingHoursPerWeek": 2.0,
        "AverageViewingDuration": 1.0,
        "ContentDownloadsPerMonth": 0,
        "UserRating": 2.0,
        "SupportTicketsPerMonth": 5,
        "WatchlistSize": 1,
    }


def test_score_customer_returns_business_fields():
    result = score_customer(DummyModel(), build_preprocessing_pipeline(), customer_row())

    assert result["prediction"] == 1
    assert result["risk_band"] == "High"
    assert result["probability"] == 0.72
    assert result["recommendation"]


def test_score_customers_requires_raw_columns():
    scored = score_customers(DummyModel(), build_preprocessing_pipeline(), pd.DataFrame([customer_row()]))

    assert [column for column in RAW_FEATURES if column not in scored.columns] == []
    assert "ChurnProbability" in scored.columns
    assert "RiskBand" in scored.columns


def test_validation_and_explainability_surface_customer_issues():
    customer = customer_row()
    customer["TotalCharges"] = 9000.0

    warnings = validate_customer(customer)
    reasons = explain_customer(customer)

    assert warnings
    assert any("Support" in reason or "support" in reason for reason in reasons)
    assert len(retention_plan("High")) >= 3
