import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine App", layout="wide", page_icon="🍷")

st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Loading Data
def load_data():
    white = pd.read_csv("winequality-white.csv", sep=";")
    red = pd.read_csv("winequality-red.csv", sep=";")
    white["type"] = "white"
    red["type"] = "red"
    return pd.concat([white, red], ignore_index=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error("CSV files not found. Please ensure 'winequality-white.csv' and 'winequality-red.csv' are in the same directory.")
    st.stop()

# Loading Models
def load_models():
    try:
        reg = joblib.load("models/regression_rf.joblib")
        clf = joblib.load("models/classification_rf.joblib")
        return reg, clf
    except FileNotFoundError:
        st.error("Model files not found. Please run 'regression_model.py' and 'classification_model.py' first.")
        return None, None

reg_model, clf_model = load_models()

if reg_model is None or clf_model is None:
    st.stop()

# Sidebar, Inputs
st.sidebar.header("Configure Wine")
st.sidebar.write("Adjust the chemical properties below:")

def user_input_features():
    with st.sidebar.expander("Acidity & Sugar", expanded=True):
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
        citric_acid = st.slider("Citric Acid", 0.0, 1.5, 0.3)
        residual_sugar = st.slider("Residual Sugar", 0.5, 65.0, 6.0)

    with st.sidebar.expander("Chemicals & Density"):
        chlorides = st.slider("Chlorides", 0.01, 0.6, 0.05)
        free_so2 = st.slider("Free SO₂", 1.0, 100.0, 30.0)
        total_so2 = st.slider("Total SO₂", 5.0, 300.0, 115.0)
        density = st.slider("Density", 0.990, 1.005, 0.996, format="%.4f")

    with st.sidebar.expander("pH, Sulphates & Alcohol"):
        pH = st.slider("pH", 2.5, 4.5, 3.2)
        sulphates = st.slider("Sulphates", 0.2, 2.0, 0.6)
        alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)
    
    wine_type = st.sidebar.radio("Select Wine Type (for Quality Model)", ["white", "red"])

    data = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_so2,
        "total sulfur dioxide": total_so2,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "type": wine_type
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Main Page
st.title("Wine Analysis")
st.markdown("### Predict Quality & Type using Random Forest Ensembles")

tabs = st.tabs([
    "Project Summary", 
    "Prediction Dashboard", 
    "Interactive EDA", 
    "Model Diagnostics"
])

# Tab 1: Summary
with tabs[0]:
    st.header("Project Overview & Dataset Summary")
    
    st.markdown("""
    This application uses Machine Learning to analyze physicochemical properties of wine.
    It combines two Random Forest models:
    1.  **Regressor:** Predicts the quality score (0-10).
    2.  **Classifier:** Predicts the wine type (Red or White).
    
    **Data Source:** [UCI Machine Learning Repository - Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)
    """)
    
    st.divider()
    
    # Dataset Statistics ---
    st.subheader("1. Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", missing_values, delta="Clean Data" if missing_values == 0 else "Dirty")
    with col4:
        st.metric("Classes", f"{df['type'].nunique()} (Red/White)")

    # Data Types DataFrame
    with st.expander("View Data Dictionary & Types"):
        dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"]).astype(str)
        dtype_df["Null Count"] = df.isnull().sum()
        st.dataframe(dtype_df, use_container_width=True)

    st.divider()
    st.info("Navigate to the **Model Diagnostics** tab to see detailed performance metrics like RMSE and R².")

# Tab 2: Prediction Dashboard
with tabs[1]:
    st.info(" <-- Adjust the wine parameters in the sidebar to generate a new prediction.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Prediction")
        predict_btn = st.button("Analyze Wine", type="primary", use_container_width=True)
        
        if predict_btn:
            with st.spinner("Analyzing chemical structure..."):
                time.sleep(0.5)
                
                pred_quality = reg_model.predict(input_df)[0]
                pred_type = clf_model.predict(input_df.drop("type", axis=1))[0]
                
                st.balloons()
                st.metric("Predicted Quality (0-10)", f"{pred_quality:.2f}", delta=f"{pred_quality-6:.2f} vs Avg")
                st.metric("Predicted Wine Type", pred_type.upper())
                
                if pred_quality > 7:
                    st.success("This is a Premium Wine!")
                elif pred_quality > 5:
                    st.info("This is a Standard Wine.")
                else:
                    st.error("Low Quality Detected.")

    with col2:
        st.subheader("Flavor Profile (Radar Chart)")
        
        categories = ['alcohol', 'pH', 'residual sugar', 'volatile acidity', 'sulphates']
        avg_wine = df[categories].mean()
        
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=input_df[categories].iloc[0].values,
            theta=categories,
            fill='toself',
            name='Your Wine',
            line_color='red'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=avg_wine.values,
            theta=categories,
            fill='toself',
            name='Average Wine',
            line_color='blue',
            opacity=0.5
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 15]) 
            ),
            showlegend=True,
            height=400,
            margin=dict(t=20, b=20, l=40, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Interactive EDA
with tabs[2]:
    st.header("Exploratory Data Analysis")
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.subheader("Alcohol vs Quality")
        fig = px.box(df, x="quality", y="alcohol", color="type", 
                     title="Alcohol Content by Quality Score")
        st.plotly_chart(fig, use_container_width=True)
        
    with col_eda2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig)

    st.subheader("Multivariate Analysis: Sugar vs Alcohol vs Quality")
    fig_scatter = px.scatter(df, x="residual sugar", y="alcohol", 
                             color="quality", size="sulphates", 
                             hover_data=['density', 'pH'],
                             title="Sugar vs Alcohol (Color=Quality, Size=Sulphates)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# Tab 4: Model Diagnostics
with tabs[3]:
    st.header("Model Diagnostics")
    st.subheader("1. Regression Model (Quality Prediction)")
    
    X_reg_full = df.drop("quality", axis=1)
    y_reg_full = df["quality"]
    
    _, X_test_reg, _, y_test_reg = train_test_split(
        X_reg_full, y_reg_full, test_size=0.2, random_state=1
    )
    
    # Predict  on the test set
    y_pred_reg = reg_model.predict(X_test_reg)
    residuals = y_test_reg - y_pred_reg

    # Calculate Metrics on Test Set
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)

    # Display Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("R² Score (Test)", f"{r2:.3f}", help="Variance explained on unseen data")
    col_m2.metric("RMSE (Test)", f"{rmse:.3f}", help="Root Mean Squared Error on unseen data")
    col_m3.metric("MAE (Test)", f"{mae:.3f}", help="Mean Absolute Error on unseen data")

    # Interpretations
    st.info(f"""
    **Metric Interpretation (Test Set):**
    These metrics are calculated on the **unseen 20% test set** to match your training script.
    * **R² Score ({r2:.2f}):** The model explains **{r2*100:.1f}%** of quality variation on new wines.
    * **RMSE ({rmse:.2f}):** Predictions are typically off by about **{rmse:.2f} points**.
    """)

    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        fig_reg = px.scatter(x=y_pred_reg, y=y_test_reg, opacity=0.3, 
                             labels={'x': 'Predicted Quality', 'y': 'Actual Quality'},
                             title="Actual vs Predicted (Test Set Only)")
        fig_reg.add_shape(type="line", x0=3, y0=3, x1=9, y1=9, line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_d2:
        fig_res = px.histogram(residuals, nbins=50, title="Distribution of Residuals (Test Set)")
        fig_res.update_layout(showlegend=False, xaxis_title="Residual Error")
        st.plotly_chart(fig_res, use_container_width=True)

    # Classification Metrics and Plots
    st.divider()
    st.subheader("2. Classification Model (Type Prediction)")
    
    X_clf_full = df.drop(["type", "quality"], axis=1)
    y_clf_full = df["type"]
    
    _, X_test_clf, _, y_test_clf = train_test_split(
        X_clf_full, y_clf_full, test_size=0.2, stratify=y_clf_full, random_state=1
    )
    
    # Predict on Test Set
    y_pred_clf = clf_model.predict(X_test_clf)
    acc = accuracy_score(y_test_clf, y_pred_clf)
    
    st.metric("Accuracy (Test Set)", f"{acc*100:.2f}%")
    
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.write("**Confusion Matrix (Test Set)**")
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["red", "white"], yticklabels=["red", "white"])
        ax.set_xlabel("Predicted Type")
        ax.set_ylabel("Actual Type")
        st.pyplot(fig_cm)
        
    with col_c2:
        try:
            importances = clf_model.named_steps['clf'].feature_importances_
            feature_names = X_clf_full.columns
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by="Importance", ascending=True)
            
            fig_imp = px.bar(feat_df, x="Importance", y="Feature", orientation='h',
                             title="Feature Importance")
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning("Could not extract feature importance directly from pipeline.")