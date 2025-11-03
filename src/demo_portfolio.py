import numpy as np
import pandas as pd


STATES = [
    "California",
    "Texas",
    "Florida",
    "New York",
    "Illinois",
    "Georgia",
    "Washington",
    "Arizona",
]

SEGMENTS = ["Consumer", "Family", "Student", "Business"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Credit Card", "Bank Withdrawal", "Mailed Check"]
CHURN_REASONS = ["Competitor", "Dissatisfaction", "Price", "Low Engagement", "Support Issues"]
INTERNET_SERVICES = ["Fiber optic", "DSL", "No"]
YES_NO = ["Yes", "No"]
CONTENT_TYPES = ["Movies", "Series", "Sports", "Kids", "Documentary"]
DEVICE_TYPES = ["Mobile", "TV", "Desktop", "Tablet"]
GENRES = ["Drama", "Comedy", "Action", "Documentary", "Sports", "Kids"]


def build_demo_portfolio(size=650, random_state=42):
    rng = np.random.default_rng(random_state)
    account_age = rng.integers(1, 84, size=size)
    monthly_charges = rng.normal(62, 21, size=size).clip(12, 145).round(2)
    viewing_hours = rng.gamma(2.4, 4.2, size=size).clip(0, 60).round(1)
    support_tickets = rng.poisson(1.2, size=size).clip(0, 9)
    user_rating = rng.normal(3.7, 0.9, size=size).clip(1, 5).round(1)
    watchlist_size = rng.integers(0, 55, size=size)
    downloads = rng.poisson(4, size=size).clip(0, 35)

    portfolio = pd.DataFrame(
        {
            "CustomerID": [f"CUST-{10000 + index}" for index in range(size)],
            "CustomerName": [f"Customer {index + 1:03d}" for index in range(size)],
            "State": rng.choice(STATES, size=size),
            "Gender": rng.choice(["Female", "Male"], size=size),
            "SeniorCitizen": rng.choice(YES_NO, size=size, p=[0.18, 0.82]),
            "Partner": rng.choice(YES_NO, size=size, p=[0.47, 0.53]),
            "Dependents": rng.choice(YES_NO, size=size, p=[0.32, 0.68]),
            "Segment": rng.choice(SEGMENTS, size=size, p=[0.48, 0.24, 0.16, 0.12]),
            "Contract": rng.choice(CONTRACTS, size=size, p=[0.58, 0.27, 0.15]),
            "PaymentMethod": rng.choice(PAYMENT_METHODS, size=size, p=[0.42, 0.38, 0.20]),
            "PaperlessBilling": rng.choice(YES_NO, size=size, p=[0.64, 0.36]),
            "ContentType": rng.choice(CONTENT_TYPES, size=size, p=[0.28, 0.31, 0.14, 0.12, 0.15]),
            "MultiDeviceAccess": rng.choice(YES_NO, size=size, p=[0.58, 0.42]),
            "DeviceRegistered": rng.choice(DEVICE_TYPES, size=size, p=[0.34, 0.30, 0.22, 0.14]),
            "AccountAge": account_age,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": (account_age * monthly_charges * rng.normal(1.0, 0.11, size=size))
            .clip(0, 10000)
            .round(2),
            "SubscriptionType": rng.choice(["Basic", "Standard", "Premium"], size=size, p=[0.44, 0.37, 0.19]),
            "ViewingHoursPerWeek": viewing_hours,
            "AverageViewingDuration": rng.normal(2.2, 1.1, size=size).clip(0, 8).round(1),
            "ContentDownloadsPerMonth": downloads,
            "GenrePreference": rng.choice(GENRES, size=size, p=[0.24, 0.18, 0.20, 0.13, 0.14, 0.11]),
            "UserRating": user_rating,
            "SupportTicketsPerMonth": support_tickets,
            "WatchlistSize": watchlist_size,
            "ParentalControl": rng.choice(YES_NO, size=size, p=[0.43, 0.57]),
            "SubtitlesEnabled": rng.choice(YES_NO, size=size, p=[0.61, 0.39]),
            "InternetService": rng.choice(INTERNET_SERVICES, size=size, p=[0.46, 0.35, 0.19]),
            "PhoneService": rng.choice(YES_NO, size=size, p=[0.9, 0.1]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=size, p=[0.44, 0.46, 0.10]),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=size, p=[0.35, 0.46, 0.19]),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=size, p=[0.42, 0.39, 0.19]),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=size, p=[0.38, 0.43, 0.19]),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=size, p=[0.34, 0.47, 0.19]),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=size, p=[0.45, 0.36, 0.19]),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=size, p=[0.44, 0.37, 0.19]),
        }
    )
    portfolio["AdminTickets"] = (
        portfolio["SupportTicketsPerMonth"] + rng.poisson(0.7, size=size)
    ).clip(0, 12)
    portfolio["TechTickets"] = (
        np.where(portfolio["TechSupport"] == "No", rng.poisson(1.3, size=size), rng.poisson(0.4, size=size))
        + portfolio["SupportTicketsPerMonth"] // 2
    ).clip(0, 12)
    portfolio["ChurnCategory"] = np.select(
        [
            portfolio["SupportTicketsPerMonth"] >= 4,
            portfolio["MonthlyCharges"] >= 95,
            portfolio["ViewingHoursPerWeek"] <= 3,
            portfolio["UserRating"] <= 2.5,
        ],
        ["Support Issues", "Price", "Low Engagement", "Dissatisfaction"],
        default=rng.choice(CHURN_REASONS, size=size),
    )
    churn_score = (
        (portfolio["Contract"] == "Month-to-month").astype(int) * 0.20
        + (portfolio["SupportTicketsPerMonth"] >= 4).astype(int) * 0.24
        + (portfolio["UserRating"] <= 2.5).astype(int) * 0.22
        + (portfolio["ViewingHoursPerWeek"] <= 3).astype(int) * 0.18
        + (portfolio["MonthlyCharges"] >= 95).astype(int) * 0.12
        + (portfolio["PaymentMethod"] == "Mailed Check").astype(int) * 0.08
    )
    portfolio["Churn"] = (churn_score >= 0.34).astype(int)

    high_risk_count = max(12, size // 8)
    high_risk_index = portfolio.index[:high_risk_count]
    portfolio.loc[high_risk_index, "Contract"] = "Month-to-month"
    portfolio.loc[high_risk_index, "PaymentMethod"] = "Mailed Check"
    portfolio.loc[high_risk_index, "MultiDeviceAccess"] = "No"
    portfolio.loc[high_risk_index, "ParentalControl"] = "No"
    portfolio.loc[high_risk_index, "SubtitlesEnabled"] = "No"
    portfolio.loc[high_risk_index, "InternetService"] = "Fiber optic"
    portfolio.loc[high_risk_index, "OnlineSecurity"] = "No"
    portfolio.loc[high_risk_index, "TechSupport"] = "No"
    portfolio.loc[high_risk_index, "PaperlessBilling"] = "Yes"
    portfolio.loc[high_risk_index, "ViewingHoursPerWeek"] = rng.uniform(0, 2.5, size=high_risk_count).round(1)
    portfolio.loc[high_risk_index, "AverageViewingDuration"] = rng.uniform(0, 0.8, size=high_risk_count).round(1)
    portfolio.loc[high_risk_index, "ContentDownloadsPerMonth"] = rng.integers(0, 2, size=high_risk_count)
    portfolio.loc[high_risk_index, "UserRating"] = rng.uniform(1.0, 2.2, size=high_risk_count).round(1)
    portfolio.loc[high_risk_index, "SupportTicketsPerMonth"] = rng.integers(5, 10, size=high_risk_count)
    portfolio.loc[high_risk_index, "AdminTickets"] = rng.integers(5, 12, size=high_risk_count)
    portfolio.loc[high_risk_index, "TechTickets"] = rng.integers(5, 12, size=high_risk_count)
    portfolio.loc[high_risk_index, "WatchlistSize"] = rng.integers(0, 3, size=high_risk_count)
    portfolio.loc[high_risk_index, "MonthlyCharges"] = rng.uniform(95, 145, size=high_risk_count).round(2)
    portfolio.loc[high_risk_index, "TotalCharges"] = (
        portfolio.loc[high_risk_index, "AccountAge"] * portfolio.loc[high_risk_index, "MonthlyCharges"]
    ).clip(0, 10000)
    portfolio.loc[high_risk_index, "ChurnCategory"] = rng.choice(
        ["Support Issues", "Price", "Low Engagement"],
        size=high_risk_count,
    )
    portfolio.loc[high_risk_index, "Churn"] = 1
    return portfolio
