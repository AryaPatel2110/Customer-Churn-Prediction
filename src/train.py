import argparse
from pathlib import Path

import cloudpickle
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from churn_features import RAW_FEATURES, build_preprocessing_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train the customer churn LightGBM model.")
    parser.add_argument("--data", required=True, help="Path to a CSV file containing training data.")
    parser.add_argument("--target", default="Churn", help="Target column name.")
    parser.add_argument("--model-out", default="smote_lgbm.pkl", help="Output path for the trained model.")
    parser.add_argument(
        "--pipeline-out",
        default="preprocessing_pipeline.pkl",
        help="Output path for the preprocessing pipeline.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_dataset(data_path, target):
    data = pd.read_csv(data_path)
    missing_columns = [column for column in [*RAW_FEATURES, target] if column not in data.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Training data is missing required columns: {missing}")

    return data[RAW_FEATURES], data[target]


def build_model(random_state):
    return LGBMClassifier(
        learning_rate=0.1,
        max_depth=20,
        n_estimators=900,
        num_leaves=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        verbosity=-1,
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return metrics, classification_report(y_test, y_pred)


def dump_artifact(path, artifact):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        cloudpickle.dump(artifact, file)


def main():
    args = parse_args()
    X, y = load_dataset(args.data, args.target)

    pipeline = build_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    X_train_resampled, y_train_resampled = SMOTE(random_state=args.random_state).fit_resample(
        X_train,
        y_train,
    )

    model = build_model(args.random_state)
    model.fit(X_train_resampled, y_train_resampled)

    metrics, report = evaluate_model(model, X_test, y_test)
    print("Classification Report")
    print(report)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    dump_artifact(args.model_out, model)
    dump_artifact(args.pipeline_out, pipeline)


if __name__ == "__main__":
    main()
