"""
Customer Churn Predictor — Flask Web App
=========================================
Author: Aryan Diwan
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, log_loss
)
import seaborn as sns

app = Flask(__name__)

BASE_DIR   = app.root_path
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(STATIC_DIR, exist_ok=True)

# ─── Load artifacts ───────────────────────────────────────────────────────────
try:
    model = XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "xgb_churn.json"))

    scaler         = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"),         "rb"))
    label_encoders = pickle.load(open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb"))
    feature_names  = pickle.load(open(os.path.join(MODEL_DIR, "feature_names.pkl"),  "rb"))

    print("✅ Model and artifacts loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


def engineer_features(df):
    """Apply the same feature engineering as training."""
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"]
    )

    df["charges_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(v not in ["No", "No internet service", "No phone service"] for v in row),
        axis=1
    )

    df["is_high_value"] = ((df["MonthlyCharges"] > df["MonthlyCharges"].median()) &
                            (df["tenure"] > 24)).astype(int)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if "Churn" in cat_cols:
        cat_cols.remove("Churn")

    for col in cat_cols:
        if col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "❌ Model not loaded.", 500

    if "file" not in request.files:
        return "❌ No file uploaded.", 400

    file = request.files["file"]
    if file.filename == "":
        return "❌ No file selected.", 400

    df = pd.read_csv(file)

    has_labels = "Churn" in df.columns
    if has_labels:
        y_true_raw = df["Churn"].copy()
        y_true = (y_true_raw == "Yes").astype(int)

    df_feat = engineer_features(df)

    # Align to training features
    missing = [f for f in feature_names if f not in df_feat.columns]
    if missing:
        return f"❌ Missing columns after engineering: {missing}", 400

    X = df_feat[feature_names]
    X_sc = scaler.transform(X)

    y_prob = model.predict_proba(X_sc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Results table
    result_df = pd.DataFrame({
        "Churn Prediction": ["Yes" if p == 1 else "No" for p in y_pred],
        "Churn Probability": [f"{p:.1%}" for p in y_prob],
        "Risk Level": ["🔴 High" if p >= 0.7 else "🟡 Medium" if p >= 0.4 else "🟢 Low" for p in y_prob]
    })

    if "customerID" in df.columns:
        result_df.insert(0, "Customer ID", df["customerID"].values)

    metrics = {}
    if has_labels:
        metrics = {
            "accuracy":  f"{accuracy_score(y_true, y_pred):.1%}",
            "precision": f"{precision_score(y_true, y_pred, zero_division=0):.1%}",
            "recall":    f"{recall_score(y_true, y_pred, zero_division=0):.1%}",
            "f1":        f"{f1_score(y_true, y_pred, zero_division=0):.1%}",
            "auc":       f"{roc_auc_score(y_true, y_prob):.4f}",
            "log_loss":  f"{log_loss(y_true, y_prob):.4f}",
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"])
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"))
        plt.close()

    # Churn distribution chart
    counts = result_df["Churn Prediction"].value_counts()
    plt.figure(figsize=(5, 4))
    plt.bar(counts.index, counts.values, color=["#00C9A7", "#FF6B6B"])
    plt.title("Predicted Churn Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "churn_distribution.png"))
    plt.close()

    # Risk distribution chart
    risk_counts = result_df["Risk Level"].value_counts()
    plt.figure(figsize=(5, 4))
    colors = {"🔴 High": "#FF6B6B", "🟡 Medium": "#FFD93D", "🟢 Low": "#00C9A7"}
    plt.bar(risk_counts.index, risk_counts.values,
            color=[colors.get(k, "#888") for k in risk_counts.index])
    plt.title("Customer Risk Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "risk_distribution.png"))
    plt.close()

    return render_template(
        "results.html",
        tables=result_df.to_html(classes="data table table-striped", index=False, escape=False),
        metrics=metrics,
        has_labels=has_labels,
        total=len(result_df),
        churn_count=int(y_pred.sum()),
        churn_rate=f"{y_pred.mean():.1%}",
    )


if __name__ == "__main__":
    app.run(debug=True)
