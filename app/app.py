import json
import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Nassau Candy Distributor.csv"
MODEL = ROOT / "models" / "nassau_slow_shipment_model.joblib"
RECS = ROOT / "results" / "final_factory_reallocation_recommendations.csv"
IMP = ROOT / "results" / "random_forest_feature_importance.csv"

FEATURES = [
    "Product Name", "Division", "Country/Region", "State/Province", "Region",
    "Factory", "Ship Mode", "Units", "Sales", "Cost", "Gross Profit",
    "Order_Year", "Order_Month", "Order_Quarter", "Order_DayOfWeek"
]

FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Kazookles": "The Other Factory",
}

TRAIN_Q3 = 1638


def canonical_product_name(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00a0", " ").strip()
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s+", " ", text)
    return text


@st.cache_data
def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    d = pd.read_csv(DATA)
    d["Order Date"] = pd.to_datetime(
        d["Order Date"], format="%d-%m-%Y", errors="raise"
    )
    d["Ship Date"] = pd.to_datetime(
        d["Ship Date"], format="%d-%m-%Y", errors="raise"
    )
    d["Delivery_Days"] = (d["Ship Date"] - d["Order Date"]).dt.days

    lookup = {
        canonical_product_name(k): v for k, v in FACTORY_MAP.items()
    }
    d["Factory"] = d["Product Name"].map(canonical_product_name).map(lookup)

    if d["Factory"].isna().any():
        missing = sorted(d.loc[d["Factory"].isna(), "Product Name"].unique())
        raise ValueError(f"Unmapped product(s): {missing}")

    d["Order_Year"] = d["Order Date"].dt.year
    d["Order_Month"] = d["Order Date"].dt.month
    d["Order_Quarter"] = d["Order Date"].dt.quarter
    d["Order_DayOfWeek"] = d["Order Date"].dt.dayofweek
    return d


@st.cache_resource
def load_model():
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")
    return joblib.load(MODEL)


@st.cache_data
def load_recommendations():
    if not RECS.exists():
        raise FileNotFoundError(f"Recommendation file not found: {RECS}")
    return pd.read_csv(RECS)


@st.cache_data
def load_importance():
    if not IMP.exists():
        raise FileNotFoundError(f"Feature-importance file not found: {IMP}")
    return pd.read_csv(IMP)


st.set_page_config(
    page_title="Nassau Candy ML Dashboard",
    page_icon="🍬",
    layout="wide",
)

st.title("🍬 Nassau Candy Factory Reallocation & Shipping Optimization")
st.caption("Machine Learning Decision-Support Dashboard | E-Commerce Analytics")

try:
    df = load_data()
    model = load_model()
    recs = load_recommendations()
    imp = load_importance()
except Exception as exc:
    st.error("Dashboard files could not be loaded.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Dashboard")
    page = st.radio(
        "Module",
        [
            "Overview",
            "Shipment Risk",
            "Factory Scenarios",
            "Model Performance",
            "Explainability",
        ],
    )
    st.divider()
    st.caption("Predictive screening prototype")

if page == "Overview":
    st.subheader("Project Overview")

    cols = st.columns(4)
    kpis = [
        ("Transactions", len(df)),
        ("Unique Orders", df["Order ID"].nunique()),
        ("Products", df["Product Name"].nunique()),
        ("Customers", df["Customer ID"].nunique()),
    ]
    for col, (name, value) in zip(cols, kpis):
        col.metric(name, f"{value:,}")

    cols = st.columns(4)
    financials = [
        ("Total Sales", f"${df['Sales'].sum():,.2f}"),
        ("Total Units", f"{df['Units'].sum():,.0f}"),
        ("Gross Profit", f"${df['Gross Profit'].sum():,.2f}"),
        (
            "GP Margin",
            f"{df['Gross Profit'].sum() / df['Sales'].sum() * 100:.2f}%",
        ),
    ]
    for col, (name, value) in zip(cols, financials):
        col.metric(name, value)

    a, b = st.columns(2)
    with a:
        st.markdown("### Shipping Mode")
        st.bar_chart(df["Ship Mode"].value_counts())
    with b:
        st.markdown("### Sales Region")
        st.bar_chart(df["Region"].value_counts())

    st.info(
        f"Slow-shipment threshold: {TRAIN_Q3} days "
        "(training-set Q3 methodology)."
    )

elif page == "Shipment Risk":
    st.subheader("🚚 Shipment Risk Prediction")
    row_index = st.number_input(
        "Historical transaction row",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )
    row = df.iloc[[int(row_index)]]

    probability = float(model.predict_proba(row[FEATURES])[:, 1][0])
    prediction = int(model.predict(row[FEATURES])[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Slow Risk", f"{probability * 100:.2f}%")
    c2.metric("Decision", "SLOW" if prediction else "NOT SLOW")
    c3.metric("Delivery Days", f"{int(row['Delivery_Days'].iloc[0])}")

    st.dataframe(
        row[
            [
                "Order ID",
                "Product Name",
                "Division",
                "Region",
                "Ship Mode",
                "Units",
                "Sales",
                "Factory",
                "Delivery_Days",
            ]
        ],
        use_container_width=True,
    )

elif page == "Factory Scenarios":
    st.subheader("🏭 Factory Reallocation Scenarios")
    st.dataframe(recs, use_container_width=True)
    st.caption(
        "Counterfactual predictive screening only; not a causal estimate "
        "or freight-cost optimizer."
    )

elif page == "Model Performance":
    st.subheader("🤖 Model Performance")
    metrics = pd.DataFrame(
        [
            ["Logistic Regression", 0.7978, 0.5070, 0.9322, 0.6568, 0.8863],
            ["Random Forest", 0.8235, 0.5457, 0.8925, 0.6773, 0.9013],
            ["Gradient Boosting", 0.8346, 0.5868, 0.6869, 0.6329, 0.9043],
        ],
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ],
    )
    st.dataframe(metrics, use_container_width=True)
    st.success(
        "Selected model: Random Forest — strongest F1/recall balance "
        "for slow-shipment screening."
    )

else:
    st.subheader("🔍 Model Explainability")
    ordered = imp.sort_values("Importance", ascending=False)
    st.dataframe(ordered, use_container_width=True)

    if {"Feature", "Importance"}.issubset(ordered.columns):
        st.bar_chart(
            ordered.head(10).set_index("Feature")["Importance"]
        )

    st.warning(
        "Impurity importance can be biased; held-out permutation importance "
        "is included in the project as a complementary diagnostic."
    )

st.divider()
st.caption(
    "No verified factory coordinates, freight rates, capacity or carrier "
    "data were available; therefore no transportation-cost savings are claimed."
)
