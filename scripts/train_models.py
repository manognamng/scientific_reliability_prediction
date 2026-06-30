import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import joblib

ROOT = Path(__file__).resolve().parents[1]

print("="*60)
print("Loading dataset...")
print("="*60)

df = pd.read_csv(ROOT/"data"/"final_dataset.csv")

print(df.shape)

# -------------------------------------------------
# Columns NOT used for ML
# -------------------------------------------------

drop_cols = [
    "pmid",
    "title_x",
    "title_y",
    "abstract",
    "node_id",
    "node_index",   # <-- add this
    "journal",
    "doi",
    "label"
]

X = df.drop(columns=drop_cols)

import numpy as np

# Replace any string representations of missing values
X = X.replace(["nan", "NaN", "None", "NULL"], np.nan)

# Fill any remaining missing values
X = X.fillna(0)

print("Remaining NaNs:", X.isna().sum().sum())
y = df["label"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)

# -------------------------------------------------
# Train/Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain:", X_train.shape)
print("Test :", X_test.shape)

# -------------------------------------------------
# Scale
# -------------------------------------------------

print("\nScaling...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    ROOT/"models"/"scaler.pkl"
)

# -------------------------------------------------
# Models
# -------------------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=2000),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric="logloss"
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        ),

    "CatBoost":
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            verbose=0,
            random_seed=42
        )
}

results = []

print("\nTraining models...\n")

for name, model in models.items():

    print("="*60)
    print(name)
    print("="*60)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test,pred)
    pre = precision_score(y_test,pred)
    rec = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    auc = roc_auc_score(y_test,prob)

    results.append([
        name,
        acc,
        pre,
        rec,
        f1,
        auc
    ])

    joblib.dump(
        model,
        ROOT/"models"/f"{name}.pkl"
    )

results = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ]
)

results = results.sort_values(
    "ROC_AUC",
    ascending=False
)

print("\n")
print(results)

results.to_csv(
    ROOT/"results"/"metrics.csv",
    index=False
)

print("\nFinished.")