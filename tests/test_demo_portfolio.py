from src.churn_features import RAW_FEATURES
from src.demo_portfolio import build_demo_portfolio


def test_demo_portfolio_contains_admin_and_model_columns():
    portfolio = build_demo_portfolio(size=25, random_state=7)

    for column in RAW_FEATURES:
        assert column in portfolio.columns

    for column in [
        "CustomerID",
        "CustomerName",
        "State",
        "Gender",
        "Partner",
        "Dependents",
        "Segment",
        "Contract",
        "PaymentMethod",
        "PaperlessBilling",
        "ContentType",
        "MultiDeviceAccess",
        "DeviceRegistered",
        "GenrePreference",
        "ParentalControl",
        "SubtitlesEnabled",
        "Churn",
        "InternetService",
        "OnlineSecurity",
        "TechSupport",
        "AdminTickets",
        "TechTickets",
    ]:
        assert column in portfolio.columns

    assert len(portfolio) == 25
