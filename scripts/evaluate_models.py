import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report
)

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT/"data"/"final_dataset.csv"

MODELS = ROOT/"models"

FIGURES = ROOT/"figures"

RESULTS = ROOT/"results"

FIGURES.mkdir(exist_ok=True)

RESULTS.mkdir(exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(DATA)

drop_cols = [
    "pmid",
    "title_x",
    "title_y",
    "abstract",
    "node_id",
    "node_index",
    "journal",
    "doi",
    "label"
]

X = df.drop(columns=drop_cols)

X = X.fillna(0)

y = df["label"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = joblib.load(
    MODELS/"scaler.pkl"
)

X_test = scaler.transform(X_test)

models = [
    "Logistic Regression",
    "Random Forest",
    "Extra Trees",
    "XGBoost",
    "LightGBM",
    "CatBoost"
]

results = []

plt.figure(figsize=(8,6))

for name in models:

    print(name)

    model = joblib.load(
        MODELS/f"{name}.pkl"
    )

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[:,1]

    results.append([
        name,
        accuracy_score(y_test,pred),
        precision_score(y_test,pred),
        recall_score(y_test,pred),
        f1_score(y_test,pred),
        roc_auc_score(y_test,prob)
    ])

    RocCurveDisplay.from_predictions(
        y_test,
        prob,
        name=name
    )

    cm = confusion_matrix(y_test,pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.savefig(
        FIGURES/f"confusion_{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    with open(
        RESULTS/"classification_report.txt",
        "a"
    ) as f:

        f.write("\n")
        f.write("="*60)
        f.write("\n")
        f.write(name)
        f.write("\n")
        f.write(classification_report(y_test,pred))
        f.write("\n")

plt.title("ROC Curves")

plt.savefig(
    FIGURES/"roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(8,6))

for name in models:

    model = joblib.load(
        MODELS/f"{name}.pkl"
    )

    prob = model.predict_proba(X_test)[:,1]

    PrecisionRecallDisplay.from_predictions(
        y_test,
        prob,
        name=name
    )

plt.title("Precision Recall Curves")

plt.savefig(
    FIGURES/"precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

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

results.to_csv(
    RESULTS/"metrics.csv",
    index=False
)

print(results)

print()

print("Done.")