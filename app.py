from html import escape
from io import StringIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.churn_features import RAW_FEATURES
from src.churn_service import (
    MODEL_METRICS,
    explain_customer,
    load_artifacts,
    model_driver_frame,
    retention_plan,
    score_customer,
    score_customers,
    validate_customer,
)


RISK_COLORS = {
    "High": "#c2410c",
    "Medium": "#b45309",
    "Low": "#15803d",
}
PRIMARY_COLOR = "#4d35e5"
SECONDARY_COLOR = "#0b1050"
LIGHT_COLOR = "#c8c2ff"
CHART_RANGE = ["#4d35e5", "#8b7cf6", "#0b1050", "#7c3aed", "#a5b4fc", "#312e81"]
ACCENT_GREEN = "#08c98f"
ACCENT_ORANGE = "#ffb703"
ACCENT_RED = "#ff5a1f"
ACCENT_BLUE = "#5967e8"
DATASET_CANDIDATES = [
    Path("data/train.csv"),
]


st.set_page_config(
    page_title="Customer Churn Risk Console",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
    }
    .dashboard-title {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .dashboard-title h1 {
        color: #0b1024;
        font-size: clamp(2.2rem, 4vw, 4.2rem);
        line-height: 0.95;
        margin: 0;
        letter-spacing: 0;
    }
    .dashboard-title p {
        color: #6b7280;
        font-size: 1.03rem;
        margin: 0.55rem 0 0;
        max-width: 780px;
    }
    .dashboard-pill {
        background: #0b1050;
        color: white;
        border-radius: 7px;
        padding: 0.75rem 1rem;
        font-size: 0.92rem;
        font-weight: 700;
        white-space: nowrap;
    }
    .summary-hero {
        background: linear-gradient(90deg, #3324c7 0%, #4d35e5 55%, #2f26b7 100%);
        border-radius: 8px;
        padding: 1.65rem 1.8rem;
        color: white;
        margin-bottom: 1.1rem;
        box-shadow: 0 18px 38px rgba(77, 53, 229, 0.2);
    }
    .hero-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0;
    }
    .hero-metric {
        min-width: 0;
        padding: 0.35rem 1.5rem;
        border-right: 1px solid rgba(255,255,255,0.34);
    }
    .hero-metric:first-child {
        padding-left: 0;
    }
    .hero-metric:last-child {
        border-right: 0;
    }
    .hero-metric h2 {
        color: white;
        font-size: clamp(2.1rem, 4.2vw, 4rem);
        line-height: 1;
        margin: 0;
        font-weight: 800;
    }
    .hero-metric p {
        color: rgba(255,255,255,0.82);
        margin: 0.55rem 0 0;
        font-size: 0.98rem;
        font-weight: 650;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(150px, 1fr));
        gap: 0.75rem;
        margin: 0.75rem 0 1.2rem;
    }
    .kpi-card {
        border: 1px solid #e7e5f4;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        background: #ffffff;
        min-height: 112px;
        box-shadow: 0 7px 20px rgba(15, 23, 42, 0.045);
        overflow-wrap: anywhere;
    }
    .kpi-card.critical {
        border-color: rgba(194, 65, 12, 0.28);
        background: #fff8f4;
    }
    .kpi-card.model {
        border-color: rgba(77, 53, 229, 0.24);
        background: #f8f7ff;
    }
    .kpi-label {
        color: #4b5563;
        font-size: 0.82rem;
        line-height: 1.2;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }
    .kpi-value {
        color: #0b1024;
        font-size: clamp(1.55rem, 2.4vw, 2.35rem);
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: 0;
    }
    .kpi-note {
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 0.45rem;
    }
    .section-label {
        background: #08c98f;
        color: white;
        border-radius: 5px;
        text-align: center;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 0.35rem 0.5rem;
        margin: 0.2rem 0 0.6rem;
        letter-spacing: 0.02rem;
    }
    .page-band {
        background: #08c98f;
        border: 1px solid #079c72;
        color: white;
        border-radius: 0;
        padding: 0.7rem 1.1rem;
        margin: 0 0 0.95rem;
        font-weight: 800;
        font-size: 1.05rem;
        letter-spacing: 0.01rem;
        text-transform: uppercase;
    }
    .analysis-panel {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        padding: 0.95rem 1rem;
        margin-bottom: 0.85rem;
    }
    .mini-stat {
        border-left: 4px solid #08c98f;
        padding: 0.3rem 0 0.55rem 0.85rem;
        margin-bottom: 0.65rem;
    }
    .mini-stat strong {
        color: #0b1024;
        display: block;
        font-size: 1.7rem;
        line-height: 1.05;
    }
    .mini-stat span {
        color: #6b7280;
        font-size: 0.8rem;
        font-weight: 650;
    }
    @media (max-width: 900px) {
        .dashboard-title {
            align-items: flex-start;
            flex-direction: column;
        }
        .hero-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            row-gap: 1.2rem;
        }
        .hero-metric:nth-child(2) {
            border-right: 0;
        }
        .kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 560px) {
        .hero-grid,
        .kpi-grid {
            grid-template-columns: 1fr;
        }
        .hero-metric {
            border-right: 0;
            border-bottom: 1px solid rgba(255,255,255,0.28);
            padding: 0.55rem 0;
        }
        .hero-metric:last-child {
            border-bottom: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model artifacts...")
def cached_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner="Scoring portfolio...")
def cached_score_portfolio(customers):
    model, pipeline, _ = cached_artifacts()
    return score_customers(model, pipeline, customers)


def format_percent(value):
    return f"{value:.1%}"


def format_currency(value):
    return f"${value:,.0f}"


def format_compact(value, prefix=""):
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{prefix}{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{prefix}{value / 1_000:.2f}K"
    return f"{prefix}{value:,.0f}"


def risk_badge(label):
    color = RISK_COLORS.get(label, "#334155")
    return (
        f"<span style='background:{color};color:white;padding:0.25rem 0.55rem;"
        f"border-radius:0.35rem;font-size:0.82rem;font-weight:700'>{label}</span>"
    )


def kpi_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def require_raw_columns(frame):
    missing = [column for column in RAW_FEATURES if column not in frame.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required model columns: {', '.join(missing)}")
        st.stop()


@st.cache_data(show_spinner="Loading project dataset...")
def load_project_dataset(path):
    return pd.read_csv(path)


def prepare_portfolio():
    dataset_path = DATASET_CANDIDATES[0]
    if not dataset_path.exists():
        st.error(
            "Dataset not found. Expected `data/train.csv`. "
            "Run `bash scripts/download_dataset.sh` from the project root."
        )
        st.stop()

    portfolio = load_project_dataset(str(dataset_path))
    require_raw_columns(portfolio)
    st.sidebar.caption(f"Dataset: {dataset_path}")
    return portfolio, str(dataset_path)


def enrich_scored_data(scored):
    enriched = scored.copy()
    enriched["RevenueAtRisk"] = enriched["MonthlyCharges"] * enriched["ChurnProbability"]
    enriched["ProbabilityLabel"] = enriched["ChurnProbability"].map(format_percent)
    enriched["ObservedChurn"] = churn_measure(enriched)
    enriched["AccountAgeGroup"] = pd.cut(
        enriched["AccountAge"],
        bins=[0, 6, 12, 24, 10_000],
        labels=["< 6 Months", "6-12 Months", "12-24 Months", ">= 24 Months"],
        include_lowest=True,
    )
    enriched["ChargeRange"] = pd.cut(
        enriched["MonthlyCharges"],
        bins=[0, 35, 65, 95, 10_000],
        labels=["Low", "Medium", "High", "Premium"],
        include_lowest=True,
    )
    return enriched


def churn_measure(frame):
    if "Churn" not in frame.columns:
        return frame["ChurnPrediction"].astype(int)

    churn = frame["Churn"]
    if pd.api.types.is_numeric_dtype(churn):
        return churn.fillna(0).astype(int).clip(0, 1)

    normalized = churn.astype(str).str.strip().str.lower()
    return normalized.isin(["1", "yes", "true", "churn", "churned"]).astype(int)


def sidebar_filters(scored):
    st.sidebar.divider()
    st.sidebar.subheader("Filters")

    filtered = scored.copy()
    for column, label in [
        ("RiskBand", "Risk band"),
        ("Segment", "Segment"),
        ("Gender", "Gender"),
        ("SubscriptionType", "Subscription"),
        ("ChargeRange", "Monthly charge"),
        ("PaymentMethod", "Payment method"),
        ("PaperlessBilling", "Paperless billing"),
        ("ContentType", "Content type"),
        ("DeviceRegistered", "Device"),
        ("GenrePreference", "Genre"),
    ]:
        if column in filtered.columns:
            options = sorted(filtered[column].dropna().unique().tolist())
            selected = st.sidebar.multiselect(label, options, default=options)
            if selected:
                filtered = filtered[filtered[column].isin(selected)]

    min_probability, max_probability = st.sidebar.slider(
        "Churn probability",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=5,
    )
    filtered = filtered[
        filtered["ChurnProbability"].between(min_probability / 100, max_probability / 100)
    ]
    return filtered


def risk_distribution(scored):
    order = ["High", "Medium", "Low"]
    counts = scored["RiskBand"].value_counts().reindex(order).fillna(0).astype(int)
    return counts


def group_risk(scored, column):
    if column not in scored.columns:
        return pd.DataFrame()
    grouped = (
        scored.groupby(column, dropna=False, observed=False)
        .agg(
            Customers=("RiskBand", "size"),
            AvgChurnProbability=("ChurnProbability", "mean"),
            HighRiskCustomers=("RiskBand", lambda values: int((values == "High").sum())),
            RevenueAtRisk=("RevenueAtRisk", "sum"),
        )
        .sort_values(["AvgChurnProbability", "HighRiskCustomers"], ascending=False)
    )
    grouped["AvgChurnProbability"] = grouped["AvgChurnProbability"].round(3)
    grouped["RevenueAtRisk"] = grouped["RevenueAtRisk"].round(0)
    return grouped


def group_churn(scored, column, limit=None):
    if column not in scored.columns:
        return pd.DataFrame()
    grouped = (
        scored.groupby(column, dropna=False, observed=False)
        .agg(
            Customers=("ObservedChurn", "size"),
            TotalChurn=("ObservedChurn", "sum"),
            ChurnRate=("ObservedChurn", "mean"),
            AvgPredictedRisk=("ChurnProbability", "mean"),
        )
        .sort_values(["ChurnRate", "TotalChurn"], ascending=False)
    )
    grouped["TotalChurn"] = grouped["TotalChurn"].astype(int)
    grouped["ChurnRate"] = grouped["ChurnRate"].round(3)
    grouped["AvgPredictedRisk"] = grouped["AvgPredictedRisk"].round(3)
    if limit:
        return grouped.head(limit)
    return grouped


def service_columns(scored):
    candidates = [
        "PaperlessBilling",
        "MultiDeviceAccess",
        "ParentalControl",
        "SubtitlesEnabled",
    ]
    return [column for column in candidates if column in scored.columns]


def yes_no_distribution(scored, columns):
    rows = []
    for column in columns:
        values = scored[column].astype(str).str.strip().str.lower()
        yes_rate = values.isin(["yes", "fiber optic", "dsl"]).mean()
        no_rate = 1 - yes_rate
        rows.append(
            {
                "Service": column,
                "No": round(no_rate, 3),
                "Yes": round(yes_rate, 3),
                "Customers": len(scored),
            }
        )
    return pd.DataFrame(rows)


def section_label(label):
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)


def show_dashboard_header():
    st.markdown(
        """
        <div class="dashboard-title">
            <div>
                <h1>Churn Analysis - Summary</h1>
                <p>Customer churn, account behavior, service adoption, content usage, and model risk.</p>
            </div>
            <div class="dashboard-pill">Churn Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_hero(total_customers, new_joiners, total_churn, churn_rate):
    st.markdown(
        f"""
        <div class="summary-hero">
            <div class="hero-grid">
                <div class="hero-metric"><h2>{total_customers:,}</h2><p>Total Customers</p></div>
                <div class="hero-metric"><h2>{new_joiners:,}</h2><p>New Joiners</p></div>
                <div class="hero-metric"><h2>{total_churn:,}</h2><p>Total Churn</p></div>
                <div class="hero-metric"><h2>{format_percent(churn_rate)}</h2><p>Churn Rate</p></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_kpi_grid(items):
    cards = []
    for item in items:
        label = escape(str(item["label"]))
        value = escape(str(item["value"]))
        note = escape(str(item.get("note", "")))
        tone = escape(str(item.get("tone", "")))
        note_html = f'<div class="kpi-note">{note}</div>' if note else ""
        cards.append(
            f'<div class="kpi-card {tone}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f"{note_html}"
            "</div>"
        )

    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def chart_frame_from_series(series, category_name="Category", value_name="Value"):
    frame = series.reset_index()
    frame.columns = [category_name, value_name]
    frame[category_name] = frame[category_name].astype(str)
    return frame


def apply_chart_theme(chart):
    return chart.configure_axis(
        labelColor="#2f3140",
        titleColor="#2f3140",
        gridColor="#eceaf8",
    ).configure_view(strokeOpacity=0)


def show_horizontal_bar(title, series, value_title="Value", percent=False):
    st.subheader(title)
    if series.empty:
        st.info("This field is not available in the current dataset.")
        return

    frame = chart_frame_from_series(series, value_name=value_title)
    tooltip_format = ".1%" if percent else ",.0f"
    axis_format = "%" if percent else "~s"
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                f"{value_title}:Q",
                axis=alt.Axis(format=axis_format),
                title=value_title,
            ),
            y=alt.Y("Category:N", sort="-x", title=None),
            color=alt.value(PRIMARY_COLOR),
            tooltip=[
                alt.Tooltip("Category:N", title="Group"),
                alt.Tooltip(f"{value_title}:Q", title=value_title, format=tooltip_format),
            ],
        )
        .properties(height=max(180, min(360, 34 * len(frame))))
    )
    st.altair_chart(apply_chart_theme(chart), use_container_width=True)


def show_donut(title, series):
    st.subheader(title)
    if series.empty:
        st.info("This field is not available in the current dataset.")
        return

    frame = chart_frame_from_series(series, value_name="Customers")
    chart = (
        alt.Chart(frame)
        .mark_arc(innerRadius=58, outerRadius=104, stroke="white", strokeWidth=2)
        .encode(
            theta=alt.Theta("Customers:Q"),
            color=alt.Color("Category:N", scale=alt.Scale(range=CHART_RANGE), legend=alt.Legend(title=None)),
            tooltip=[
                alt.Tooltip("Category:N", title="Group"),
                alt.Tooltip("Customers:Q", title="Customers", format=","),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(apply_chart_theme(chart), use_container_width=True)


def show_combo_chart(title, frame):
    st.subheader(title)
    if frame.empty:
        st.info("This field is not available in the current dataset.")
        return

    chart_data = frame.reset_index()
    category = chart_data.columns[0]
    chart_data[category] = chart_data[category].astype(str)

    bars = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{category}:N", sort=None, title=None),
            y=alt.Y("Customers:Q", title="Total Customers"),
            color=alt.value(PRIMARY_COLOR),
            tooltip=[
                alt.Tooltip(f"{category}:N", title="Group"),
                alt.Tooltip("Customers:Q", title="Customers", format=","),
                alt.Tooltip("TotalChurn:Q", title="Churned", format=","),
                alt.Tooltip("ChurnRate:Q", title="Churn Rate", format=".1%"),
            ],
        )
    )
    line = (
        alt.Chart(chart_data)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), color=SECONDARY_COLOR, strokeWidth=3)
        .encode(
            x=alt.X(f"{category}:N", sort=None, title=None),
            y=alt.Y("ChurnRate:Q", title="Churn Rate", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip(f"{category}:N", title="Group"),
                alt.Tooltip("ChurnRate:Q", title="Churn Rate", format=".1%"),
            ],
        )
    )
    chart = alt.layer(bars, line).resolve_scale(y="independent").properties(height=290)
    st.altair_chart(apply_chart_theme(chart), use_container_width=True)


def show_churn_rate_chart(title, frame, limit=None):
    if limit and not frame.empty:
        frame = frame.head(limit)
    series = frame["ChurnRate"] if not frame.empty else pd.Series(dtype=float)
    show_horizontal_bar(title, series, value_title="Churn Rate", percent=True)


def show_churn_count_chart(title, series):
    show_horizontal_bar(title, series, value_title="Total Churn")


def show_stacked_service_chart(title, frame):
    st.subheader(title)
    if frame.empty:
        st.info("This field is not available in the current dataset.")
        return

    chart_data = frame.reset_index().rename(columns={"index": "Service"})
    chart_data = chart_data.melt(
        id_vars=["Service"],
        value_vars=["No", "Yes"],
        var_name="Adoption",
        value_name="Share",
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Customer Share"),
            y=alt.Y("Service:N", sort="-x", title=None),
            color=alt.Color(
                "Adoption:N",
                scale=alt.Scale(domain=["No", "Yes"], range=[LIGHT_COLOR, PRIMARY_COLOR]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Service:N"),
                alt.Tooltip("Adoption:N"),
                alt.Tooltip("Share:Q", format=".1%"),
            ],
        )
        .properties(height=max(230, 28 * frame.shape[0]))
    )
    st.altair_chart(apply_chart_theme(chart), use_container_width=True)


def show_scatter(title, frame, x_column, y_column, color_column="RiskBand"):
    st.subheader(title)
    if frame.empty or x_column not in frame.columns or y_column not in frame.columns:
        st.info("This field is not available in the current dataset.")
        return

    chart = (
        alt.Chart(frame)
        .mark_circle(size=70, opacity=0.68)
        .encode(
            x=alt.X(f"{x_column}:Q", title=x_column),
            y=alt.Y(f"{y_column}:Q", title=y_column, axis=alt.Axis(format="%" if y_column == "ChurnProbability" else "")),
            color=alt.Color(
                f"{color_column}:N",
                scale=alt.Scale(
                    domain=["High", "Medium", "Low"],
                    range=[RISK_COLORS["High"], RISK_COLORS["Medium"], RISK_COLORS["Low"]],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("CustomerID:N", title="Customer ID") if "CustomerID" in frame.columns else alt.Tooltip(f"{x_column}:Q"),
                alt.Tooltip(f"{x_column}:Q", format=".2f"),
                alt.Tooltip(f"{y_column}:Q", format=".1%" if y_column == "ChurnProbability" else ".2f"),
                alt.Tooltip(f"{color_column}:N"),
            ],
        )
        .properties(height=285)
    )
    st.altair_chart(apply_chart_theme(chart), use_container_width=True)


def show_churn_table(title, frame):
    st.subheader(title)
    if frame.empty:
        st.info("This field is not available in the current dataset.")
    else:
        st.dataframe(frame, use_container_width=True)


def show_page_band(title):
    st.markdown(f"<div class='page-band'>{escape(title)}</div>", unsafe_allow_html=True)


def show_mini_stat(value, label):
    st.markdown(
        f"<div class='mini-stat'><strong>{escape(str(value))}</strong><span>{escape(str(label))}</span></div>",
        unsafe_allow_html=True,
    )


def yes_share(scored, column):
    if column not in scored.columns or scored.empty:
        return 0
    return scored[column].astype(str).str.strip().str.lower().eq("yes").mean()


def category_share(scored, column, value):
    if column not in scored.columns or scored.empty:
        return 0
    return scored[column].astype(str).str.strip().str.lower().eq(str(value).lower()).mean()


def show_service_tiles(scored):
    st.subheader("Subscribed Services")
    tiles = [
        ("Multi Device", yes_share(scored, "MultiDeviceAccess")),
        ("Parental Control", yes_share(scored, "ParentalControl")),
        ("Subtitles", yes_share(scored, "SubtitlesEnabled")),
        ("Paperless Billing", yes_share(scored, "PaperlessBilling")),
        ("Premium Plan", category_share(scored, "SubscriptionType", "Premium")),
        ("Basic Plan", category_share(scored, "SubscriptionType", "Basic")),
    ]
    show_kpi_grid(
        [
            {
                "label": label,
                "value": format_percent(value),
                "note": "Customer adoption share",
                "tone": "model" if index % 2 else "",
            }
            for index, (label, value) in enumerate(tiles)
        ]
    )


def show_sum_rate_combo(title, scored, group_column, sum_column):
    st.subheader(title)
    if group_column not in scored.columns or sum_column not in scored.columns:
        st.info("This field is not available in the current dataset.")
        return

    chart_data = (
        scored.groupby(group_column, observed=False)
        .agg(
            ChurnRate=("ObservedChurn", "mean"),
            TotalValue=(sum_column, "sum"),
        )
        .sort_values("ChurnRate", ascending=False)
        .reset_index()
    )
    chart_data[group_column] = chart_data[group_column].astype(str)

    bars = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{group_column}:N", sort=None, title=None),
            y=alt.Y("ChurnRate:Q", title="Churn Rate", axis=alt.Axis(format="%")),
            color=alt.value(ACCENT_ORANGE),
            tooltip=[
                alt.Tooltip(f"{group_column}:N", title="Group"),
                alt.Tooltip("ChurnRate:Q", title="Churn Rate", format=".1%"),
                alt.Tooltip("TotalValue:Q", title=sum_column, format=",.0f"),
            ],
        )
    )
    line = (
        alt.Chart(chart_data)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), color=SECONDARY_COLOR, strokeWidth=3)
        .encode(
            x=alt.X(f"{group_column}:N", sort=None, title=None),
            y=alt.Y("TotalValue:Q", title=sum_column, axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("TotalValue:Q", title=sum_column, format=",.0f")],
        )
    )
    chart = alt.layer(bars, line).resolve_scale(y="independent").properties(height=285)
    st.altair_chart(apply_chart_theme(chart), use_container_width=True)


def show_customer_churn_dashboard(scored):
    show_page_band("Customer Churn Dashboard")

    customers_at_risk = int(scored["ObservedChurn"].sum())
    support_tickets = int(scored["SupportTicketsPerMonth"].sum())
    yearly_charges = scored["TotalCharges"].sum()
    monthly_charges = scored["MonthlyCharges"].sum()
    high_risk = int((scored["RiskBand"] == "High").sum())

    show_kpi_grid(
        [
            {"label": "Customers At Risk", "value": f"{customers_at_risk:,}", "note": "Observed churned customers", "tone": "critical"},
            {"label": "Support Tickets", "value": f"{support_tickets:,}", "note": "Monthly support tickets"},
            {"label": "High Risk Customers", "value": f"{high_risk:,}", "note": "Model probability >= 50%", "tone": "model"},
            {"label": "Yearly Charges", "value": format_compact(yearly_charges, "$"), "note": "TotalCharges sum"},
            {"label": "Monthly Charges", "value": format_compact(monthly_charges, "$"), "note": "MonthlyCharges sum"},
        ]
    )

    section_label("DEMOGRAPHICS BY GENDER")
    left, middle, right = st.columns([1.05, 0.55, 1])
    with left:
        show_donut(
            "Churned Customers By Gender",
            scored.groupby("Gender")["ObservedChurn"].sum().sort_values(ascending=False),
        )
    with middle:
        show_mini_stat(format_percent(yes_share(scored, "ParentalControl")), "Parental Control")
        show_mini_stat(format_percent(yes_share(scored, "MultiDeviceAccess")), "Multi Device Access")
        show_mini_stat(format_percent(yes_share(scored, "SubtitlesEnabled")), "Subtitles Enabled")
    with right:
        tenure_share = scored["AccountAgeGroup"].value_counts(normalize=True).sort_index()
        show_horizontal_bar("Subscription Time", tenure_share, value_title="Customer Share", percent=True)

    section_label("CUSTOMER ACCOUNT INFORMATION")
    left, middle, right = st.columns([1, 0.7, 1])
    with left:
        payment_share = scored["PaymentMethod"].value_counts(normalize=True)
        show_horizontal_bar("Payment Method", payment_share, value_title="Customer Share", percent=True)
    with middle:
        show_donut(
            "Paperless Billing",
            scored["PaperlessBilling"].value_counts(),
        )
        show_mini_stat(format_currency(scored["MonthlyCharges"].mean()), "Avg Monthly Charge")
        show_mini_stat(format_currency(scored["TotalCharges"].mean()), "Avg Total Charges")
    with right:
        show_churn_rate_chart("Churn Rate By Subscription", group_churn(scored, "SubscriptionType"))

    section_label("SUBSCRIBED SERVICES")
    left, right = st.columns([1, 1])
    with left:
        show_service_tiles(scored)
    with right:
        show_horizontal_bar(
            "Content Type Users",
            scored["ContentType"].value_counts(normalize=True),
            value_title="Customer Share",
            percent=True,
        )
        show_horizontal_bar(
            "Device Registered",
            scored["DeviceRegistered"].value_counts(normalize=True),
            value_title="Customer Share",
            percent=True,
        )


def show_customer_risk_dashboard(scored):
    show_page_band("Customer Risk Analysis Dashboard")

    total_customers = len(scored)
    churn_rate = scored["ObservedChurn"].mean() if total_customers else 0
    yearly_charges = scored["TotalCharges"].sum()
    support_tickets = int(scored["SupportTicketsPerMonth"].sum())
    revenue_at_risk = scored["RevenueAtRisk"].sum()

    show_kpi_grid(
        [
            {"label": "Total Customers", "value": f"{total_customers:,}", "note": "Filtered customer base"},
            {"label": "Churn Rate", "value": format_percent(churn_rate), "note": "Observed churn rate", "tone": "critical"},
            {"label": "Yearly Charges", "value": format_compact(yearly_charges, "$"), "note": "TotalCharges sum"},
            {"label": "Support Tickets", "value": f"{support_tickets:,}", "note": "Total support tickets"},
            {"label": "Revenue At Risk", "value": format_compact(revenue_at_risk, "$"), "note": "Monthly charge weighted by model risk", "tone": "model"},
        ]
    )

    left, middle, right = st.columns([1, 1, 1])
    with left:
        show_churn_rate_chart("Churn Rate By Subscription Type", group_churn(scored, "SubscriptionType"))
    with middle:
        show_donut("# Of Customers By Subscription", scored["SubscriptionType"].value_counts())
    with right:
        show_donut("Sum Of Monthly Charges", scored.groupby("SubscriptionType")["MonthlyCharges"].sum())

    left, middle, right = st.columns([1, 1, 1])
    with left:
        show_combo_chart("Type Of Plan Risk", group_churn(scored, "SubscriptionType"))
    with middle:
        show_sum_rate_combo("Monthly Charge Range Risk", scored, "ChargeRange", "MonthlyCharges")
    with right:
        show_sum_rate_combo("Churn By Payment Method", scored, "PaymentMethod", "MonthlyCharges")

    left, right = st.columns([1, 1])
    with left:
        show_scatter(
            "Viewing Hours Vs Churn Probability",
            scored.sample(min(len(scored), 5000), random_state=42) if len(scored) > 5000 else scored,
            "ViewingHoursPerWeek",
            "ChurnProbability",
        )
    with right:
        show_scatter(
            "Support Tickets Vs Churn Probability",
            scored.sample(min(len(scored), 5000), random_state=7) if len(scored) > 5000 else scored,
            "SupportTicketsPerMonth",
            "ChurnProbability",
        )


def show_executive_dashboard(scored):
    show_dashboard_header()

    total_customers = len(scored)
    new_joiners = int((scored["AccountAge"] <= 6).sum()) if total_customers else 0
    total_churn = int(scored["ObservedChurn"].sum()) if total_customers else 0
    churn_rate = scored["ObservedChurn"].mean() if total_customers else 0
    avg_probability = scored["ChurnProbability"].mean() if total_customers else 0
    revenue_at_risk = scored["RevenueAtRisk"].sum()

    show_hero(total_customers, new_joiners, total_churn, churn_rate)

    high_risk = int((scored["RiskBand"] == "High").sum())
    support_avg = scored["SupportTicketsPerMonth"].mean() if not scored.empty else 0
    viewing_avg = scored["ViewingHoursPerWeek"].mean() if not scored.empty else 0
    rating_avg = scored["UserRating"].mean() if not scored.empty else 0
    downloads_avg = scored["ContentDownloadsPerMonth"].mean() if not scored.empty else 0
    payment_risk = group_churn(scored, "PaymentMethod")
    highest_risk_payment = str(payment_risk.index[0]) if not payment_risk.empty else "N/A"

    high_risk_share = high_risk / total_customers if total_customers else 0
    observed_source = "Actual churn column" if "Churn" in scored.columns else "Model prediction"
    show_kpi_grid(
        [
            {
                "label": "High Risk Customers",
                "value": f"{high_risk:,}",
                "note": f"{format_percent(high_risk_share)} of filtered customers",
                "tone": "critical",
            },
            {
                "label": "Avg Predicted Risk",
                "value": format_percent(avg_probability),
                "note": "Mean model churn probability",
                "tone": "model",
            },
            {
                "label": "Revenue At Risk",
                "value": format_currency(revenue_at_risk),
                "note": "Monthly charge weighted by risk",
                "tone": "critical",
            },
            {
                "label": "Avg Support Tickets",
                "value": f"{support_avg:.1f}",
                "note": "Monthly support load",
            },
            {
                "label": "Avg Viewing Hours",
                "value": f"{viewing_avg:.1f}",
                "note": "Weekly engagement",
            },
            {
                "label": "Avg User Rating",
                "value": f"{rating_avg:.1f}",
                "note": "Customer satisfaction signal",
            },
            {
                "label": "Avg Downloads",
                "value": f"{downloads_avg:.1f}",
                "note": "Content downloads per month",
            },
            {
                "label": "Avg Monthly Charge",
                "value": format_currency(scored["MonthlyCharges"].mean()),
                "note": "Current monthly billing",
            },
            {
                "label": "Avg Total Charges",
                "value": format_currency(scored["TotalCharges"].mean()),
                "note": "Customer lifetime billing",
            },
            {
                "label": "Highest Churn Payment",
                "value": highest_risk_payment,
                "note": "Payment method with highest churn rate",
                "tone": "model",
            },
            {
                "label": "Churn Source",
                "value": observed_source,
                "note": "Used for dashboard churn rates",
            },
        ]
    )

    section_label("DEMOGRAPHIC")
    left, middle, right = st.columns([1, 1.15, 1])
    with left:
        show_donut(
            "Total Churn By Gender",
            scored.groupby("Gender")["ObservedChurn"].sum().sort_values(ascending=False)
            if "Gender" in scored.columns
            else pd.Series(dtype=float),
        )
    with middle:
        show_combo_chart(
            "Customers And Churn Rate By Account Age",
            group_churn(scored, "AccountAgeGroup"),
        )
    with right:
        show_churn_rate_chart("Churn Rate By Parental Control", group_churn(scored, "ParentalControl"))

    section_label("ACCOUNT INFO")
    left, middle, right = st.columns([1, 1, 1])
    with left:
        show_churn_rate_chart(
            "Churn Rate By Payment Method",
            group_churn(scored, "PaymentMethod"),
        )
        show_churn_rate_chart(
            "Churn Rate By Paperless Billing",
            group_churn(scored, "PaperlessBilling"),
        )
    with middle:
        show_combo_chart(
            "Customers And Churn Rate By Tenure Group",
            group_churn(scored, "AccountAgeGroup"),
        )
    with right:
        show_churn_rate_chart(
            "Churn Rate By Monthly Charge",
            group_churn(scored, "ChargeRange"),
        )

    section_label("SERVICES USED")
    left, middle, right = st.columns([1, 1, 1])
    with left:
        show_churn_rate_chart(
            "Churn Rate By Subscription",
            group_churn(scored, "SubscriptionType"),
        )
        show_churn_rate_chart(
            "Churn Rate By Multi Device Access",
            group_churn(scored, "MultiDeviceAccess"),
        )
    with middle:
        show_churn_rate_chart(
            "Churn Rate By Content Type",
            group_churn(scored, "ContentType"),
        )
        show_churn_rate_chart(
            "Churn Rate By Device Registered",
            group_churn(scored, "DeviceRegistered"),
        )
    with right:
        services = yes_no_distribution(scored, service_columns(scored))
        show_stacked_service_chart("Services Adoption", services.set_index("Service") if not services.empty else services)

    section_label("ENGAGEMENT AND MODEL RISK")
    left, middle, right = st.columns([1, 1, 1])
    with left:
        show_donut("Risk Band Distribution", risk_distribution(scored))
    with middle:
        revenue_by_band = (
            scored.groupby("RiskBand")["RevenueAtRisk"]
            .sum()
            .reindex(["High", "Medium", "Low"])
            .fillna(0)
        )
        show_horizontal_bar("Revenue At Risk By Band", revenue_by_band, value_title="Revenue At Risk")
    with right:
        show_churn_rate_chart(
            "Churn Rate By Genre Preference",
            group_churn(scored, "GenrePreference"),
        )

    left, right = st.columns([1, 1])
    with left:
        show_scatter(
            "Viewing Hours Vs Predicted Churn Risk",
            scored,
            "ViewingHoursPerWeek",
            "ChurnProbability",
        )
    with right:
        show_scatter(
            "Support Tickets Vs Predicted Churn Risk",
            scored,
            "SupportTicketsPerMonth",
            "ChurnProbability",
        )

    section_label("STAKEHOLDER ACTION LIST")
    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Highest Priority Customers")
        high_priority_columns = [
            column
            for column in [
                "CustomerID",
                "Gender",
                "AccountAge",
                "SubscriptionType",
                "PaymentMethod",
                "PaperlessBilling",
                "ContentType",
                "DeviceRegistered",
                "MonthlyCharges",
                "Churn",
                "ChurnProbability",
                "RiskBand",
                "RevenueAtRisk",
                "RecommendedAction",
            ]
            if column in scored.columns
        ]
        st.dataframe(
            scored.sort_values(["ChurnProbability", "RevenueAtRisk"], ascending=False)
            .head(25)[high_priority_columns],
            use_container_width=True,
        )
    with right:
        st.subheader("What Stakeholders Should Do")
        st.write(f"- Investigate `{highest_risk_payment}` as the payment method with the highest churn rate.")
        st.write("- Prioritize high-risk customers with high monthly revenue exposure.")
        st.write("- Compare payment method and contract churn rates before offering discounts.")
        st.write("- Use service/content cuts to decide whether engagement or support intervention is best.")


def show_customer_explorer(scored):
    st.title("Customer Risk Explorer")
    st.caption("Search, sort, and export the customers that need retention action.")

    search = st.text_input("Search customer ID or name").strip().lower()
    table = scored.copy()
    if search:
        searchable = pd.Series(False, index=table.index)
        for column in ["CustomerID", "CustomerName"]:
            if column in table.columns:
                searchable = searchable | table[column].astype(str).str.lower().str.contains(search)
        table = table[searchable]

    sort_by = st.selectbox(
        "Sort customers by",
        ["ChurnProbability", "RevenueAtRisk", "MonthlyCharges", "SupportTicketsPerMonth", "UserRating"],
    )
    table = table.sort_values(sort_by, ascending=sort_by == "UserRating")

    summary_cols = st.columns(4)
    with summary_cols[0]:
        kpi_card("Shown Customers", f"{len(table):,}")
    with summary_cols[1]:
        kpi_card("High Risk Shown", f"{int((table['RiskBand'] == 'High').sum()):,}")
    with summary_cols[2]:
        kpi_card("Shown Revenue Risk", format_currency(table["RevenueAtRisk"].sum()))
    with summary_cols[3]:
        avg = table["ChurnProbability"].mean() if len(table) else 0
        kpi_card("Shown Avg Risk", format_percent(avg))

    columns = [
        column
        for column in [
            "CustomerID",
            "CustomerName",
            "Segment",
            "PaymentMethod",
            "SubscriptionType",
            "MonthlyCharges",
            "ChurnProbability",
            "RiskBand",
            "RevenueAtRisk",
            "SupportTicketsPerMonth",
            "ViewingHoursPerWeek",
            "UserRating",
            "WatchlistSize",
            "RecommendedAction",
        ]
        if column in table.columns
    ]
    st.dataframe(table[columns].head(250), use_container_width=True)

    csv_buffer = StringIO()
    table.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download scored customers",
        csv_buffer.getvalue(),
        file_name="scored_customers.csv",
        mime="text/csv",
    )

    st.subheader("Customer Detail")
    customer_labels = []
    if "CustomerID" in table.columns:
        customer_labels = table["CustomerID"].astype(str).head(250).tolist()

    if not customer_labels:
        st.warning("No customers match the current filters.")
        return

    selected_id = st.selectbox("Select customer", customer_labels)
    selected = table[table["CustomerID"].astype(str) == selected_id].iloc[0]
    customer = selected[RAW_FEATURES].to_dict()
    reasons = explain_customer(customer)
    warnings = validate_customer(customer)
    plan = retention_plan(selected["RiskBand"])

    left, right = st.columns([1, 2])
    with left:
        st.markdown(risk_badge(selected["RiskBand"]), unsafe_allow_html=True)
        st.metric("Churn Probability", format_percent(selected["ChurnProbability"]))
        st.metric("Monthly Revenue", format_currency(selected["MonthlyCharges"]))
        st.metric("Revenue At Risk", format_currency(selected["RevenueAtRisk"]))
    with right:
        st.write("Risk reasons")
        for reason in reasons:
            st.write(f"- {reason}")
        if warnings:
            st.warning(" ".join(warnings))
        st.write("Retention plan")
        for item in plan:
            st.write(f"- {item}")


def numeric_input(label, value, min_value=0.0, max_value=None, step=1.0):
    kwargs = {"label": label, "value": value, "min_value": min_value, "step": step}
    if max_value is not None:
        kwargs["max_value"] = max_value
    return st.number_input(**kwargs)


def show_prediction_page():
    st.title("Prediction Webpage")
    st.caption("Score one customer and generate an immediate retention recommendation.")

    with st.form("single_prediction_form"):
        left, middle, right = st.columns(3)
        with left:
            account_age = st.number_input("Account Age", min_value=1, value=12, step=1)
            subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"], index=1)
            monthly_charges = numeric_input("Monthly Charges", 62.0, 0.0, None, 1.0)
            total_charges = numeric_input("Total Charges", 744.0, 0.0, None, 10.0)
        with middle:
            viewing_hours = numeric_input("Viewing Hours Per Week", 8.0, 0.0, 80.0, 0.5)
            avg_duration = numeric_input("Average Viewing Duration", 2.0, 0.0, 12.0, 0.25)
            downloads = st.number_input("Content Downloads Per Month", min_value=0, value=4, step=1)
        with right:
            rating = numeric_input("User Rating", 3.8, 1.0, 5.0, 0.1)
            tickets = st.number_input("Support Tickets Per Month", min_value=0, value=1, step=1)
            watchlist = st.number_input("Watchlist Size", min_value=0, value=8, step=1)

        submitted = st.form_submit_button("Score Customer", type="primary")

    customer = {
        "AccountAge": int(account_age),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "SubscriptionType": subscription,
        "ViewingHoursPerWeek": float(viewing_hours),
        "AverageViewingDuration": float(avg_duration),
        "ContentDownloadsPerMonth": int(downloads),
        "UserRating": float(rating),
        "SupportTicketsPerMonth": int(tickets),
        "WatchlistSize": int(watchlist),
    }

    if not submitted:
        st.info("Enter customer details and score the customer.")
        return

    model, pipeline, _ = cached_artifacts()
    result = score_customer(model, pipeline, customer)
    reasons = explain_customer(customer)
    plan = retention_plan(result["risk_band"])

    left, right = st.columns([1, 2])
    with left:
        st.markdown(risk_badge(result["risk_band"]), unsafe_allow_html=True)
        prediction_label = "Likely to churn" if result["prediction"] == 1 else "Likely to retain"
        st.metric("Prediction", prediction_label)
        st.metric("Churn Probability", format_percent(result["probability"]))
    with right:
        st.subheader("Recommended Action")
        st.write(result["recommendation"])
        if result["warnings"]:
            st.warning(" ".join(result["warnings"]))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Risk Explanation")
        for reason in reasons:
            st.write(f"- {reason}")
    with right:
        st.subheader("Retention Playbook")
        for item in plan:
            st.write(f"- {item}")


def show_model_monitoring():
    st.title("Model Monitoring")
    st.caption("Model artifact health, training metrics, and global drivers.")

    model, _, pipeline_source = cached_artifacts()
    columns = st.columns(5)
    with columns[0]:
        kpi_card("Model", MODEL_METRICS["Model"])
    with columns[1]:
        kpi_card("Accuracy", MODEL_METRICS["Accuracy"])
    with columns[2]:
        kpi_card("ROC-AUC", MODEL_METRICS["ROC-AUC"])
    with columns[3]:
        kpi_card("Runtime", MODEL_METRICS["Runtime"])
    with columns[4]:
        kpi_card("Pipeline", pipeline_source)

    drivers = model_driver_frame(model, limit=15)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Global Feature Importance")
        if drivers.empty:
            st.warning("Feature importance is unavailable for this model artifact.")
        else:
            chart = (
                alt.Chart(drivers)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Importance:Q", title="Importance"),
                    y=alt.Y("Feature:N", sort="-x", title=None),
                    color=alt.value(PRIMARY_COLOR),
                    tooltip=[
                        alt.Tooltip("Feature:N"),
                        alt.Tooltip("Importance:Q", format=","),
                    ],
                )
                .properties(height=420)
            )
            st.altair_chart(apply_chart_theme(chart), use_container_width=True)
    with right:
        st.subheader("Operational Risk Bands")
        st.write("High: churn probability greater than or equal to 50%")
        st.write("Medium: churn probability from 20% to 49.9%")
        st.write("Low: churn probability below 20%")
        st.subheader("Model Inputs")
        st.write(", ".join(RAW_FEATURES))


def main():
    model, pipeline, pipeline_source = cached_artifacts()
    _ = model, pipeline

    page = st.sidebar.radio(
        "Page",
        [
            "Customer Churn Dashboard",
            "Customer Risk Analysis Dashboard",
            "Customer Risk Explorer",
            "Prediction Webpage",
            "Model Monitoring",
        ],
    )
    st.sidebar.caption(f"Pipeline source: {pipeline_source}")

    if page in {"Customer Churn Dashboard", "Customer Risk Analysis Dashboard", "Customer Risk Explorer"}:
        portfolio, _ = prepare_portfolio()
        scored = enrich_scored_data(cached_score_portfolio(portfolio))
        filtered = sidebar_filters(scored)

        if filtered.empty:
            st.warning("No customers match the current filters.")
            return

        if page == "Customer Churn Dashboard":
            show_customer_churn_dashboard(filtered)
        elif page == "Customer Risk Analysis Dashboard":
            show_customer_risk_dashboard(filtered)
        else:
            show_customer_explorer(filtered)
    elif page == "Prediction Webpage":
        show_prediction_page()
    else:
        show_model_monitoring()


if __name__ == "__main__":
    main()
