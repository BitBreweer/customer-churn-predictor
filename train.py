"""
Customer Churn Prediction — Training Pipeline
==============================================
Dataset : IBM Telco Customer Churn (Kaggle)
Model   : XGBoost + SHAP explainability
Author  : Aryan Diwan
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
from xgboost import XGBClassifier
import shap

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("📦 Loading data...")
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(f"   Shape: {df.shape}")
print(f"   Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")

# ─────────────────────────────────────────────
# 2. CLEANING
# ─────────────────────────────────────────────
print("\n🧹 Cleaning...")

# TotalCharges has spaces as missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customerID — not a feature
df.drop(columns=["customerID"], inplace=True)

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n⚙️  Engineering features...")

# Tenure groups
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 60, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"]
)

# Charge per month ratio
df["charges_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

# Number of services subscribed
service_cols = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]
df["num_services"] = df[service_cols].apply(
    lambda row: sum(v not in ["No", "No internet service", "No phone service"] for v in row),
    axis=1
)

# High value customer flag
df["is_high_value"] = ((df["MonthlyCharges"] > df["MonthlyCharges"].median()) &
                        (df["tenure"] > 24)).astype(int)

print(f"   Total features after engineering: {df.shape[1]}")

# ─────────────────────────────────────────────
# 4. ENCODING
# ─────────────────────────────────────────────
print("\n🔢 Encoding...")

label_encoders = {}
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "Churn"]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Target
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

feature_names = X.columns.tolist()

# ─────────────────────────────────────────────
# 6. MODEL — XGBoost
# ─────────────────────────────────────────────
print("\n🤖 Training XGBoost...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # handle imbalance
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train_sc, y_train,
    eval_set=[(X_test_sc, y_test)],
    verbose=50
)

# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
print("\n📊 Evaluating...")

y_pred  = model.predict(X_test_sc)
y_prob  = model.predict_proba(X_test_sc)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)

print(f"\n   Accuracy  : {acc:.4f}")
print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec:.4f}")
print(f"   F1 Score  : {f1:.4f}")
print(f"   ROC-AUC   : {auc:.4f}")

cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="roc_auc")
print(f"   5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────
import os
os.makedirs("static", exist_ok=True)

# Confusion matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, lw=2, label=f"XGBoost (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/roc_curve.png")
plt.close()

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_names)
top15 = importances.nlargest(15)
plt.figure(figsize=(8, 6))
top15.sort_values().plot(kind="barh", color="#00C9A7")
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.close()

# SHAP summary plot
print("\n🔍 Computing SHAP values...")
explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_test_sc)
plt.figure()
shap.summary_plot(shap_vals, X_test_sc, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("static/shap_summary.png", bbox_inches="tight")
plt.close()

print("✅ All plots saved to static/")

# ─────────────────────────────────────────────
# 9. SAVE ARTIFACTS
# ─────────────────────────────────────────────
print("\n💾 Saving model artifacts...")
os.makedirs("model", exist_ok=True)

model.save_model("model/xgb_churn.json")

for name, obj in [("scaler", scaler), ("label_encoders", label_encoders)]:
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

with open("model/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("✅ Saved: model/xgb_churn.json, scaler.pkl, label_encoders.pkl, feature_names.pkl")
print("\n🏁 Training complete!")
