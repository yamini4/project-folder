import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Adult Income Classification", layout="wide")
st.title("📊 Adult Income – Model Evaluation App - ML Classification Models")

# ---------------- LOAD MODELS ----------------
MODEL_MAP = {
    "Logistic Regression": pickle.load(open("model/logistic_reg_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("model/decision_tree_model.pkl", "rb")),
    "KNN": pickle.load(open("model/KNN_model.pkl", "rb")),
    "Naive Bayes": pickle.load(open("model/naive_bayes_model.pkl", "rb")),
    "Random Forest": pickle.load(open("model/random_forest_model.pkl", "rb")),
    "XGBoost": pickle.load(open("model/XGBoost_model.pkl", "rb"))
}

model_choice = st.selectbox("Select a Model", MODEL_MAP.keys())

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Test Dataset (CSV only)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully")

    if "income" not in data.columns:
        st.error("❌ Target column 'income' not found in dataset")
        st.stop()

    # ---------------- PREPROCESSING ----------------
    data.columns = data.columns.str.strip()

    # Clean target
    data["income"] = data["income"].str.strip()
    data["income"] = LabelEncoder().fit_transform(data["income"])

    # Encode categorical columns
    categorical_cols = data.select_dtypes(include="object").columns
    for col in categorical_cols:
        data[col] = data[col].str.strip()
        data[col] = LabelEncoder().fit_transform(data[col])

    X = data.drop("income", axis=1)
    y = data["income"]

    # ---------------- LOAD SELECTED MODEL ----------------
    model = MODEL_MAP[model_choice]

    # ---------------- PREDICTION ----------------
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        y_prob = None
        auc = "Not Available"

    # ---------------- METRICS ----------------
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

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

    report_dict = classification_report(
        y,
        y_pred,
        target_names=["<=50K", ">50K"],
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose().round(3)

    st.dataframe(report_df)
