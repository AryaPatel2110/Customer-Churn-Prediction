"""Export the trained churn system as compact assets for static hosting."""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.api import get_scored_portfolio, model, model_info, portfolio_summary
from src.churn_features import RAW_FEATURES


OUTPUT_DIR = BASE_DIR / "frontend" / "public" / "data"


def compact_tree(node):
    if "leaf_value" in node:
        return node["leaf_value"]
    return [
        node["split_feature"],
        node["threshold"],
        1 if node["default_left"] else 0,
        compact_tree(node["left_child"]),
        compact_tree(node["right_child"]),
    ]


def compact_customer_rows(frame):
    return [
        [
            row.CustomerID,
            row.SubscriptionType,
            row.PaymentMethod,
            row.ContentType,
            row.DeviceRegistered,
            int(row.AccountAge),
            round(float(row.MonthlyCharges), 4),
            round(float(row.ViewingHoursPerWeek), 4),
            round(float(row.UserRating), 4),
            int(row.SupportTicketsPerMonth),
            round(float(row.ChurnProbability), 8),
            int(row.ChurnPrediction),
            row.RiskBand,
            int(row.ContentDownloadsPerMonth),
            int(row.WatchlistSize),
        ]
        for row in frame.itertuples(index=False)
    ]


def write_json(name, payload):
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"{path.relative_to(BASE_DIR)}: {path.stat().st_size / 1024 / 1024:.2f} MiB")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = get_scored_portfolio()
    booster = model.booster_.dump_model()

    write_json("summary.json", portfolio_summary())
    write_json("model-info.json", model_info())
    write_json(
        "customers.json",
        {
            "columns": [
                "customer_id",
                "subscription_type",
                "payment_method",
                "content_type",
                "device",
                "account_age",
                "monthly_charges",
                "viewing_hours",
                "user_rating",
                "support_tickets",
                "probability",
                "prediction",
                "risk_band",
                "downloads",
                "watchlist_size",
            ],
            "rows": compact_customer_rows(frame),
        },
    )
    write_json(
        "predictor.json",
        {
            "model": "LightGBM + SMOTE",
            "objective": booster["objective"],
            "feature_names": booster["feature_names"],
            "trees": [compact_tree(tree["tree_structure"]) for tree in booster["tree_info"]],
            "verification": [
                {
                    "customer": {
                        feature: (
                            int(row[feature])
                            if feature
                            in {
                                "AccountAge",
                                "ContentDownloadsPerMonth",
                                "SupportTicketsPerMonth",
                                "WatchlistSize",
                            }
                            else row[feature]
                        )
                        for feature in RAW_FEATURES
                    },
                    "probability": float(row["ChurnProbability"]),
                }
                for _, row in frame.head(8).iterrows()
            ],
        },
    )


if __name__ == "__main__":
    main()
