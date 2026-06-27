from functools import lru_cache
from pathlib import Path
from typing import List, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.churn_features import MODEL_FEATURES, RAW_FEATURES
from src.churn_service import (
    MODEL_METRICS,
    explain_customer,
    load_artifacts,
    model_driver_frame,
    retention_plan,
    score_customer,
    score_customers,
)


BASE_DIR = Path(__file__).resolve().parents[1]
TEST_DATA_PATH = BASE_DIR / "data" / "test.csv"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"


class CustomerPayload(BaseModel):
    AccountAge: int = Field(..., ge=1)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    SubscriptionType: Literal["Basic", "Standard", "Premium"]
    ViewingHoursPerWeek: float = Field(..., ge=0)
    AverageViewingDuration: float = Field(..., ge=0)
    ContentDownloadsPerMonth: int = Field(..., ge=0)
    UserRating: float = Field(..., ge=1, le=5)
    SupportTicketsPerMonth: int = Field(..., ge=0)
    WatchlistSize: int = Field(..., ge=0)


app = FastAPI(
    title="Customer Churn Intelligence API",
    version="2.0.0",
    description="Scores existing customers and powers the customer churn analytics dashboard.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, pipeline, pipeline_source = load_artifacts()


@lru_cache(maxsize=1)
def get_scored_portfolio():
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Existing-customer dataset not found at {TEST_DATA_PATH}")
    customers = pd.read_csv(TEST_DATA_PATH)
    return score_customers(model, pipeline, customers)


def grouped_risk(frame, column):
    grouped = (
        frame.groupby(column, dropna=False)
        .agg(
            customers=("CustomerID", "count"),
            predicted_churn=("ChurnPrediction", "sum"),
            average_risk=("ChurnProbability", "mean"),
        )
        .reset_index()
        .sort_values("average_risk", ascending=False)
    )
    grouped[column] = grouped[column].astype(str)
    return grouped.to_dict(orient="records")


def customer_record(row):
    raw = {feature: row[feature] for feature in RAW_FEATURES}
    return {
        "customer_id": row["CustomerID"],
        "subscription_type": row["SubscriptionType"],
        "payment_method": row["PaymentMethod"],
        "content_type": row["ContentType"],
        "device": row["DeviceRegistered"],
        "account_age": int(row["AccountAge"]),
        "monthly_charges": float(row["MonthlyCharges"]),
        "viewing_hours": float(row["ViewingHoursPerWeek"]),
        "user_rating": float(row["UserRating"]),
        "support_tickets": int(row["SupportTicketsPerMonth"]),
        "probability": float(row["ChurnProbability"]),
        "prediction": int(row["ChurnPrediction"]),
        "risk_band": row["RiskBand"],
        "recommended_action": row["RecommendedAction"],
        "reasons": explain_customer(raw),
        "retention_plan": retention_plan(row["RiskBand"]),
    }


@app.get("/api/health")
@app.get("/health", include_in_schema=False)
def health():
    return {
        "status": "ok",
        "model": MODEL_METRICS["Model"],
        "pipeline_source": pipeline_source,
        "portfolio_source": "data/test.csv",
    }


@app.get("/api/portfolio/summary")
def portfolio_summary():
    try:
        frame = get_scored_portfolio()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    counts = frame["RiskBand"].value_counts()
    top_risk = frame.nlargest(6, "ChurnProbability")
    return {
        "total_customers": int(len(frame)),
        "predicted_churn": int(frame["ChurnPrediction"].sum()),
        "predicted_churn_rate": float(frame["ChurnPrediction"].mean()),
        "average_probability": float(frame["ChurnProbability"].mean()),
        "high_risk": int(counts.get("High", 0)),
        "medium_risk": int(counts.get("Medium", 0)),
        "low_risk": int(counts.get("Low", 0)),
        "monthly_revenue": float(frame["MonthlyCharges"].sum()),
        "revenue_at_risk": float(
            (frame["MonthlyCharges"] * frame["ChurnProbability"]).sum()
        ),
        "risk_distribution": [
            {"name": band, "value": int(counts.get(band, 0))}
            for band in ("High", "Medium", "Low")
        ],
        "subscription_risk": grouped_risk(frame, "SubscriptionType"),
        "payment_risk": grouped_risk(frame, "PaymentMethod"),
        "content_risk": grouped_risk(frame, "ContentType"),
        "top_risk_customers": [customer_record(row) for _, row in top_risk.iterrows()],
    }


@app.get("/api/portfolio/customers")
def portfolio_customers(
    search: str = "",
    risk: Literal["All", "High", "Medium", "Low"] = "All",
    subscription: Literal["All", "Basic", "Standard", "Premium"] = "All",
    sort: Literal["risk_desc", "risk_asc", "charges_desc", "tenure_desc"] = "risk_desc",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    frame = get_scored_portfolio()
    filtered = frame
    if search:
        filtered = filtered[
            filtered["CustomerID"].astype(str).str.contains(search, case=False, na=False)
        ]
    if risk != "All":
        filtered = filtered[filtered["RiskBand"] == risk]
    if subscription != "All":
        filtered = filtered[filtered["SubscriptionType"] == subscription]

    sort_map = {
        "risk_desc": ("ChurnProbability", False),
        "risk_asc": ("ChurnProbability", True),
        "charges_desc": ("MonthlyCharges", False),
        "tenure_desc": ("AccountAge", False),
    }
    sort_column, ascending = sort_map[sort]
    filtered = filtered.sort_values(sort_column, ascending=ascending)
    start = (page - 1) * page_size
    rows = filtered.iloc[start : start + page_size]
    return {
        "items": [customer_record(row) for _, row in rows.iterrows()],
        "total": int(len(filtered)),
        "page": page,
        "page_size": page_size,
        "pages": max(1, (len(filtered) + page_size - 1) // page_size),
    }


@app.get("/api/portfolio/customers/{customer_id}")
def portfolio_customer(customer_id: str):
    frame = get_scored_portfolio()
    match = frame[frame["CustomerID"] == customer_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer_record(match.iloc[0])


@app.get("/api/model")
def model_info():
    drivers = model_driver_frame(model, limit=10)
    return {
        "metrics": MODEL_METRICS,
        "pipeline_source": pipeline_source,
        "raw_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "drivers": [
            {"feature": row["Feature"], "importance": int(row["Importance"])}
            for _, row in drivers.iterrows()
        ],
        "training_rows": 243787,
        "training_churn_rate": 0.18123197709475894,
        "workflow": [
            "Raw customer signals",
            "Feature engineering",
            "Categorical encoding",
            "Feature selection",
            "LightGBM prediction",
            "Retention action",
        ],
    }


@app.post("/api/predict")
@app.post("/predict", include_in_schema=False)
def predict(customer: CustomerPayload):
    result = score_customer(model, pipeline, customer.model_dump())
    result["reasons"] = explain_customer(customer.model_dump())
    result["retention_plan"] = retention_plan(result["risk_band"])
    return result


@app.post("/api/predict-batch")
@app.post("/predict-batch", include_in_schema=False)
def predict_batch(customers: List[CustomerPayload]):
    customer_frame = pd.DataFrame([customer.model_dump() for customer in customers])
    scored = score_customers(model, pipeline, customer_frame)
    return scored.to_dict(orient="records")


if FRONTEND_DIST.exists():
    assets_path = FRONTEND_DIST / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_react_app(full_path: str):
        requested = FRONTEND_DIST / full_path
        if full_path and requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
