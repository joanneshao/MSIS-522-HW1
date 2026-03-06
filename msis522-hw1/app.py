import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="MSIS 522 Housing App", layout="wide")

st.title("California Housing Price Prediction")
st.write("MSIS 522 Homework 1: End-to-End Data Science Workflow")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    housing_data = fetch_california_housing(as_frame=True)
    return housing_data.frame

df = load_data()

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    lr_model = joblib.load("models/linear_regression.joblib")
    dt_model = joblib.load("models/decision_tree_best.joblib")
    rf_model = joblib.load("models/random_forest_best.joblib")
    xgb_model = joblib.load("models/xgboost_best.joblib")
    return lr_model, dt_model, rf_model, xgb_model

lr_model, dt_model, rf_model, xgb_model = load_models()

# -----------------------------
# Load model comparison results
# -----------------------------
results_df = pd.read_csv("model_comparison_results.csv")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction"
])

# =========================================================
# Tab 1
# =========================================================
with tab1:
    st.header("Executive Summary")

    st.write("""
    This project uses the California Housing dataset to predict median house value at the district level.
    The task is a regression problem, and the target variable is **MedHouseVal**.
    """)

    st.write("""
    This problem matters because accurate housing price prediction supports real estate analysis,
    investment decisions, urban planning, and affordability assessment.
    """)

    st.write("""
    I followed a complete data science workflow including descriptive analytics, predictive modeling,
    model comparison, SHAP-based explainability, and deployment through a Streamlit web application.
    """)

    best_model = results_df.sort_values("RMSE").iloc[0]["Model"]
    st.success(f"Best-performing model by RMSE: {best_model}")

# =========================================================
# Tab 2
# =========================================================
with tab2:
    st.header("Descriptive Analytics")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Target Distribution")
    st.image("artifacts/target_distribution.png", caption="Distribution of the target variable MedHouseVal.")

    st.subheader("Feature Relationships")
    st.image("artifacts/income_vs_value.png", caption="Median income is positively associated with house value.")

    st.subheader("Correlation Heatmap")
    st.image("artifacts/correlation_heatmap.png", caption="Correlation structure among numerical variables.")

# =========================================================
# Tab 3
# =========================================================
with tab3:
    st.header("Model Performance")

    st.subheader("Model Comparison Table")
    st.dataframe(results_df)

    st.subheader("RMSE Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results_df["Model"], results_df["RMSE"])
    ax.set_ylabel("RMSE")
    ax.set_title("Model Comparison by RMSE")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.subheader("Predicted vs Actual Plot")
    st.image("artifacts/rf_actual_vs_pred.png", caption="Random Forest predicted vs actual values.")

    st.subheader("Best Hyperparameters")
    if "Best Params" in results_df.columns:
        st.dataframe(results_df[["Model", "Best Params"]])
    else:
        st.write("Best hyperparameters are not available in the results file.")

# =========================================================
# Tab 4
# =========================================================
with tab4:
    st.header("Explainability & Interactive Prediction")

    st.subheader("SHAP Explainability")
    st.image("artifacts/shap_summary.png", caption="SHAP summary plot.")
    st.image("artifacts/shap_bar.png", caption="SHAP feature importance bar plot.")
    st.image("artifacts/shap_waterfall.png", caption="SHAP waterfall plot for a sample prediction.")

    st.subheader("Interactive Prediction")

    model_choice = st.selectbox(
        "Choose a model",
        ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
    )

    medinc = st.slider("Median Income (MedInc)", 0.0, 15.0, 4.0)
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

    st.success(f"Predicted median house value: {pred:.3f}")