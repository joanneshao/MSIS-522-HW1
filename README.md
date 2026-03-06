[README.md](https://github.com/user-attachments/files/25803457/README.md)
# California Housing Price Prediction (MSIS 522)

This project builds and evaluates several machine learning models to predict housing prices using the California Housing dataset.

## Dataset
The dataset comes from `sklearn.datasets.fetch_california_housing`.

Target variable:
- **MedHouseVal** — median house value.

Features include:
- Median income
- House age
- Average number of rooms
- Population
- Latitude / Longitude

## Machine Learning Models

The following models were trained and compared:

- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- Neural Network (MLP)

Evaluation metrics include:

- MAE
- RMSE
- R²

## Explainability

Model explainability was implemented using **SHAP**:

- SHAP summary plot
- SHAP feature importance bar chart
- SHAP waterfall plot

These visualizations explain how features influence predictions.

## Streamlit Application

A Streamlit app was built to present the full workflow:

Tabs include:

1. Executive Summary
2. Descriptive Analytics
3. Model Performance
4. Explainability & Interactive Prediction

The app allows users to:

- explore dataset visualizations
- compare model performance
- generate predictions interactively
- view SHAP explanations

## Deployment

The app is deployed using **Streamlit Community Cloud**.
