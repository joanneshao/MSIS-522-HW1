import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import fetch_california_housing
from pathlib import Path

st.set_page_config(page_title="MSIS 522 Housing App", layout="wide")

st.title("California Housing Price Prediction")
st.write("MSIS 522 Homework 1: End-to-End Data Science Workflow")

# ------------------------------------------------
# Define base directory (important for deployment)
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# ------------------------------------------------
# Load data
# ------------------------------------------------
@st.cache_data
def load_data():
    housing_data = fetch_california_housing(as_frame=True)
    return housing_data.frame

df = load_data()

# ------------------------------------------------
# Load models
# ------------------------------------------------
@st.cache_resource
def load_models():
    lr_model = joblib.load(MODELS_DIR / "linear_regression.joblib")
    dt_model = joblib.load(MODELS_DIR / "decision_tree_best.joblib")
    rf_model = joblib.load(MODELS_DIR / "random_forest_best.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_best.joblib")
    return lr_model, dt_model, rf_model, xgb_model

lr_model, dt_model, rf_model, xgb_model = load_models()

# ------------------------------------------------
# Load model comparison results
# ------------------------------------------------
results_df = pd.read_csv(BASE_DIR / "model_comparison_results.csv")

# ------------------------------------------------
# Tabs
# ------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction"
])

# =========================================================
# Tab 1 — Executive Summary
# =========================================================
with tab1:

    st.header("Executive Summary")

    st.write("""
    This project uses the California Housing dataset to predict median house value at the district level.
    The task is a regression problem and the target variable is **MedHouseVal**.
    """)

    st.write("""
    Accurate housing price prediction is important for real estate investment, urban planning,
    and housing market analysis.
    """)

    st.write("""
    This project demonstrates an end-to-end data science workflow including:

    • Exploratory data analysis  
    • Machine learning model development  
    • Model comparison  
    • SHAP explainability  
    • Streamlit deployment
    """)

    best_model = results_df.sort_values("RMSE").iloc[0]["Model"]

    st.success(f"Best-performing model by RMSE: {best_model}")

# =========================================================
# Tab 2 — Descriptive Analytics
# =========================================================
with tab2:

    st.header("Descriptive Analytics")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Target Distribution")

    st.image(
        ARTIFACTS_DIR / "target_distribution.png",
        caption="Distribution of Median House Value"
    )

    st.subheader("Income vs Housing Value")

    st.image(
        ARTIFACTS_DIR / "income_vs_value.png",
        caption="Median Income is positively correlated with housing value"
    )

    st.subheader("Correlation Heatmap")

    st.image(
        ARTIFACTS_DIR / "correlation_heatmap.png",
        caption="Correlation matrix of numerical features"
    )

# =========================================================
# Tab 3 — Model Performance
# =========================================================
with tab3:

    st.header("Model Performance")

    st.subheader("Model Comparison Table")

    st.dataframe(results_df)

    st.subheader("RMSE Comparison")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(results_df["Model"], results_df["RMSE"])

    ax.set_ylabel("RMSE")

    ax.set_title("Model Comparison by RMSE")

    plt.xticks(rotation=30)

    st.pyplot(fig)

    st.subheader("Predicted vs Actual Plot (Random Forest)")

    st.image(
        ARTIFACTS_DIR / "rf_actual_vs_pred.png",
        caption="Random Forest predicted vs actual values"
    )

# =========================================================
# Tab 4 — Explainability & Interactive Prediction
# =========================================================
with tab4:

    st.header("Explainability & Interactive Prediction")

    st.subheader("SHAP Explainability")

    st.image(ARTIFACTS_DIR / "shap_summary.png")
    st.image(ARTIFACTS_DIR / "shap_bar.png")
    st.image(ARTIFACTS_DIR / "shap_waterfall.png")

    st.subheader("Interactive Prediction")

    model_choice = st.selectbox(
        "Choose a model",
        ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
    )

    medinc = st.slider("Median Income", 0.0, 15.0, 4.0)
    houseage = st.slider("House Age", 1.0, 60.0, 30.0)
    averooms = st.slider("Average Rooms", 1.0, 15.0, 5.0)
    avebedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
    population = st.slider("Population", 1.0, 10000.0, 1500.0)
    aveoccup = st.slider("Average Occupancy", 0.5, 10.0, 3.0)
    latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
    longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

    input_df = pd.DataFrame([{
        "MedInc": medinc,
        "HouseAge": houseage,
        "AveRooms": averooms,
        "AveBedrms": avebedrms,
        "Population": population,
        "AveOccup": aveoccup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    if model_choice == "Linear Regression":
        pred = lr_model.predict(input_df)[0]

    elif model_choice == "Decision Tree":
        pred = dt_model.predict(input_df)[0]

    elif model_choice == "Random Forest":
        pred = rf_model.predict(input_df)[0]

    elif model_choice == "XGBoost":
        pred = xgb_model.predict(input_df)[0]

    st.success(f"Predicted Median House Value: {pred:.3f}")