from typing import List, Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.churn_service import MODEL_METRICS, load_artifacts, score_customer, score_customers


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
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="API for scoring customer churn risk from account, billing, support, and engagement signals.",
)

model, pipeline, pipeline_source = load_artifacts()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_METRICS["Model"],
        "pipeline_source": pipeline_source,
    }


@app.post("/predict")
def predict(customer: CustomerPayload):
    return score_customer(model, pipeline, customer.model_dump())


@app.post("/predict-batch")
def predict_batch(customers: List[CustomerPayload]):
    customer_frame = pd.DataFrame([customer.model_dump() for customer in customers])
    scored = score_customers(model, pipeline, customer_frame)
    return scored.to_dict(orient="records")
