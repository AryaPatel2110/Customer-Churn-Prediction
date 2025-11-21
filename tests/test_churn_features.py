import pandas as pd

from src.churn_features import MODEL_FEATURES, build_preprocessing_pipeline


def test_preprocessing_pipeline_creates_expected_model_features():
    customer = pd.DataFrame(
        [
            {
                "AccountAge": 10,
                "MonthlyCharges": 40.0,
                "TotalCharges": 400.0,
                "SubscriptionType": "Premium",
                "ViewingHoursPerWeek": 8.0,
                "AverageViewingDuration": 2.0,
                "ContentDownloadsPerMonth": 3,
                "UserRating": 4.0,
                "SupportTicketsPerMonth": 1,
                "WatchlistSize": 4,
            }
        ]
    )

    transformed = build_preprocessing_pipeline().fit_transform(customer)

    assert list(transformed.columns) == MODEL_FEATURES
    assert transformed.loc[0, "SubscriptionType"] == 2
    assert transformed.loc[0, "TotalSpendingRatio"] == 40.0
    assert transformed.loc[0, "ViewingIntensity"] == 2.0
