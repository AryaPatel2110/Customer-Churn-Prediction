import argparse
from pathlib import Path

import cloudpickle
import pandas as pd

from churn_features import RAW_FEATURES


def parse_args():
    parser = argparse.ArgumentParser(description="Score customer churn from a CSV file.")
    parser.add_argument("--input", required=True, help="Path to a CSV file with customer rows.")
    parser.add_argument("--output", help="Optional CSV path for scored rows.")
    parser.add_argument("--model", default="smote_lgbm.pkl", help="Path to the trained model artifact.")
    parser.add_argument(
        "--pipeline",
        default="preprocessing_pipeline.pkl",
        help="Path to the preprocessing pipeline artifact.",
    )
    return parser.parse_args()


def load_artifact(path):
    with Path(path).open("rb") as file:
        return cloudpickle.load(file)


def main():
    args = parse_args()
    model = load_artifact(args.model)
    pipeline = load_artifact(args.pipeline)

    customers = pd.read_csv(args.input)
    missing_columns = [column for column in RAW_FEATURES if column not in customers.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input data is missing required columns: {missing}")

    processed_customers = pipeline.transform(customers[RAW_FEATURES])
    scored_customers = customers.copy()
    scored_customers["ChurnPrediction"] = model.predict(processed_customers)
    scored_customers["ChurnProbability"] = model.predict_proba(processed_customers)[:, 1]

    if args.output:
        scored_customers.to_csv(args.output, index=False)
    else:
        print(scored_customers.to_csv(index=False))


if __name__ == "__main__":
    main()
