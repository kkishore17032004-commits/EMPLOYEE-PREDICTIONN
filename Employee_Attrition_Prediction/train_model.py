"""
Train and save the Employee Attrition Prediction model bundle.
Run: python train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "attrition_model.pkl"
DATASET_CANDIDATES = (
    "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    "DATASET.csv",
)


def resolve_dataset_path() -> Path:
    """Return the first existing HR dataset CSV in the project folder."""
    for name in DATASET_CANDIDATES:
        path = BASE_DIR / name
        if path.exists():
            return path
    parent = BASE_DIR.parent / "DATASET.csv"
    if parent.exists():
        return parent
    raise FileNotFoundError(
        f"No dataset found in {BASE_DIR}. Add WA_Fn-UseC_-HR-Employee-Attrition.csv or DATASET.csv"
    )


def train_and_save() -> None:
    """Train all models and persist the best model bundle."""
    df = pd.read_csv(resolve_dataset_path())
    df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], inplace=True)
    df_raw = df.copy()
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    comparison = []
    evaluation = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        comparison.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1 Score": f1_score(y_test, pred, zero_division=0),
            }
        )
        fpr, tpr, _ = roc_curve(y_test, proba)
        evaluation[name] = {
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))},
        }
        trained[name] = model

    comparison.sort(key=lambda x: x["Accuracy"], reverse=True)
    best_name = comparison[0]["Model"]

    cat_options = {col: sorted(df_raw[col].unique().tolist()) for col in cat_cols}
    num_ranges = {
        col: {
            "min": int(df_raw[col].min()),
            "max": int(df_raw[col].max()),
            "mean": float(df_raw[col].mean()),
        }
        for col in feature_columns
        if col not in cat_cols
    }

    bundle = {
        "model": trained[best_name],
        "best_model_name": best_name,
        "feature_columns": feature_columns,
        "label_encoders": encoders,
        "categorical_columns": cat_cols,
        "categorical_options": cat_options,
        "numerical_ranges": num_ranges,
        "model_comparison": comparison,
        "all_models": trained,
        "evaluation": evaluation,
        "accuracy": comparison[0]["Accuracy"],
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"Best model: {best_name} ({comparison[0]['Accuracy'] * 100:.2f}%)")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
