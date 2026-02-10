# -*- coding: utf-8 -*-
"""
Improved Insurance Cost Prediction Model
Feature Engineering enhanced version
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Create required directories
# ------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 70)
print("IMPROVED INSURANCE COST PREDICTION MODEL")
print("Feature Engineering Enhanced")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv("data/insurance.csv")
print(f"Dataset loaded: {df.shape}")

# ------------------------------------------------------------------
# 2. One-Hot Encoding
# ------------------------------------------------------------------
df_encoded = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True
)

# ------------------------------------------------------------------
# 3. Feature Engineering
# ------------------------------------------------------------------
df_encoded["bmi_smoker"] = df_encoded["bmi"] * df_encoded["smoker_yes"]
df_encoded["is_obese"] = (df_encoded["bmi"] > 30).astype(int)

df_encoded["age_group"] = pd.cut(
    df_encoded["age"],
    bins=[0, 25, 35, 50, 100],
    labels=[0, 1, 2, 3]
).astype(int)

df_encoded["bmi_category"] = pd.cut(
    df_encoded["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=[0, 1, 2, 3]
).astype(int)

df_encoded["high_risk"] = (
    (df_encoded["smoker_yes"] == 1) &
    (df_encoded["is_obese"] == 1)
).astype(int)

df_encoded["age_bmi"] = df_encoded["age"] * df_encoded["bmi"]
df_encoded["has_children"] = (df_encoded["children"] > 0).astype(int)

# ------------------------------------------------------------------
# 4. Train / Test split
# ------------------------------------------------------------------
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 5. Baseline model (no FE)
# ------------------------------------------------------------------
df_base = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True
)

X_base = df_base.drop("charges", axis=1)
y_base = df_base["charges"]

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_base, y_base, test_size=0.2, random_state=42
)

baseline_model = LinearRegression()
baseline_model.fit(Xb_train, yb_train)
baseline_pred = baseline_model.predict(Xb_test)

baseline_r2 = r2_score(yb_test, baseline_pred)

# ------------------------------------------------------------------
# 6. Models with Feature Engineering
# ------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results[name] = {
        "model": model,
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds))
    }

# ------------------------------------------------------------------
# 7. Select best model
# ------------------------------------------------------------------
best_model_name = max(results, key=lambda k: results[k]["r2"])
best_model = results[best_model_name]["model"]

improvement = (
    (results[best_model_name]["r2"] - baseline_r2)
    / baseline_r2
) * 100

print(f"Best model: {best_model_name}")
print(f"Improvement: +{improvement:.2f}%")

# ------------------------------------------------------------------
# 8. Save artifacts (joblib)
# ------------------------------------------------------------------
joblib.dump(best_model, "models/best_model_with_fe.joblib")

joblib.dump(
    {
        "feature_names": X.columns.tolist(),
        "model_type": best_model_name,
        "r2_score": results[best_model_name]["r2"]
    },
    "models/model_info_fe.joblib"
)

joblib.dump(
    {
        "original_r2": baseline_r2,
        "best_r2": results[best_model_name]["r2"],
        "improvement": improvement
    },
    "models/comparison_results.joblib"
)

print("Models and metadata saved successfully.")
