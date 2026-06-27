from src.api import CustomerPayload, model_info, portfolio_customers, portfolio_summary, predict


def test_portfolio_summary_scores_existing_customers():
    summary = portfolio_summary()

    assert summary["total_customers"] == 104480
    assert sum(item["value"] for item in summary["risk_distribution"]) == 104480
    assert summary["high_risk"] == summary["predicted_churn"]
    assert len(summary["top_risk_customers"]) == 6


def test_customer_list_and_single_prediction_contract():
    customers = portfolio_customers(risk="High", page=1, page_size=5)
    assert customers["total"] > 0
    assert all(customer["risk_band"] == "High" for customer in customers["items"])

    result = predict(
        CustomerPayload(
            AccountAge=18,
            MonthlyCharges=34,
            TotalCharges=612,
            SubscriptionType="Basic",
            ViewingHoursPerWeek=3,
            AverageViewingDuration=28,
            ContentDownloadsPerMonth=1,
            UserRating=2.4,
            SupportTicketsPerMonth=5,
            WatchlistSize=2,
        )
    )
    assert 0 <= result["probability"] <= 1
    assert result["risk_band"] in {"Low", "Medium", "High"}
    assert result["reasons"]
    assert result["retention_plan"]


def test_model_metadata_matches_pipeline_contract():
    metadata = model_info()

    assert len(metadata["raw_features"]) == 10
    assert len(metadata["model_features"]) == 15
    assert metadata["metrics"]["Model"] == "LightGBM + SMOTE"
