import streamlit as st
import pandas as pd
import numpy as np
import importlib
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ML Classification App", layout="wide")
st.title("📊 ML Classification Models – Evaluation App")

# ---------------- MODEL SELECTION ----------------
MODEL_MAP = {
    "Logistic Regression": "logistic_reg_model",
    "Decision Tree": "decision_tree_model",
    "KNN": "KNN_model",
    "Naive Bayes": "naive_bayes_model",
    "Random Forest": "random_forest_model",
    "XGBoost": "XGBoost_model"
}

model_choice = st.selectbox("Select a Model", MODEL_MAP.keys())

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV only)",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully")

    if "income" not in data.columns:
        st.error("❌ Target column 'income' not found in dataset")
        st.stop()

    X = data.drop("income", axis=1)
    y = data["income"]

    # ---------------- LOAD MODEL ----------------
    try:
        module_name = MODEL_MAP[model_choice]
        model_module = importlib.import_module(f"model.{module_name}")
        model = model_module.load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    # ---------------- PREDICTION ----------------
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = None

    # ---------------- METRICS ----------------
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    if y_prob is not None:
        auc = roc_auc_score(y, y_prob)
    else:
        auc = "Not Available"

    # ---------------- DISPLAY METRICS ----------------
    st.subheader("📌 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(acc, 4))
    col2.metric("Precision", round(prec, 4))
    col3.metric("Recall", round(rec, 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1, 4))
    col5.metric("MCC", round(mcc, 4))
    col6.metric("AUC", auc if isinstance(auc, str) else round(auc, 4))

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("📉 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ---------------- CLASSIFICATION REPORT ----------------
    st.subheader("📄 Classification Report")
    st.text(classification_report(y, y_pred))
