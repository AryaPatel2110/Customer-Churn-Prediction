import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


RAW_FEATURES = [
    "AccountAge",
    "MonthlyCharges",
    "TotalCharges",
    "SubscriptionType",
    "ViewingHoursPerWeek",
    "AverageViewingDuration",
    "ContentDownloadsPerMonth",
    "UserRating",
    "SupportTicketsPerMonth",
    "WatchlistSize",
]

MODEL_FEATURES = [
    "ContentConsumptionScore",
    "AccountAge",
    "AverageViewingDuration",
    "TotalCharges",
    "SupportEngagementRatio",
    "ViewingHoursPerWeek",
    "CustomerTenureEngagement",
    "MonthlyCharges",
    "TotalSpendingRatio",
    "UserRating",
    "ContentDownloadsPerMonth",
    "ViewingIntensity",
    "WatchlistSize",
    "SupportTicketsPerMonth",
    "SubscriptionType",
]


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        account_age = X["AccountAge"].replace(0, np.nan)
        viewing_hours = X["ViewingHoursPerWeek"].replace(0, np.nan)
        watchlist_size = X["WatchlistSize"].replace(0, np.nan)

        X["TotalSpendingRatio"] = (X["TotalCharges"] / account_age).fillna(0)
        X["MonthlyChangeInCharges"] = X["MonthlyCharges"] - X["TotalSpendingRatio"]
        X["ViewingIntensity"] = (X["ViewingHoursPerWeek"] / watchlist_size).fillna(0)
        X["SupportEngagementRatio"] = (X["SupportTicketsPerMonth"] / viewing_hours).fillna(0)
        X["ContentConsumptionScore"] = (
            X["ViewingHoursPerWeek"]
            + X["AverageViewingDuration"]
            + X["ContentDownloadsPerMonth"]
        )
        X["CustomerTenureEngagement"] = (X["AccountAge"] / viewing_hours).fillna(0)
        return X


class CategoricalEncoding(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        subscription_mapping = {"Basic": 0, "Standard": 1, "Premium": 2}
        X["SubscriptionType"] = X["SubscriptionType"].map(subscription_mapping).fillna(-1)
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features=None):
        self.selected_features = selected_features or MODEL_FEATURES

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.selected_features]


def build_preprocessing_pipeline():
    return Pipeline(
        steps=[
            ("features", FeatureEngineering()),
            ("categorical", CategoricalEncoding()),
            ("selector", FeatureSelector()),
        ]
    )
