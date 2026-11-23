"""Reproducible Bagging / Boosting / Stacking comparison."""
# AI-assisted implementation: OpenAI ChatGPT (GPT-6), 2026-11-22. See AI_DISCLOSURE.md.

from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FEATURES
BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"


def preprocess():
    return ColumnTransformer(transformers=[
        ("numbers", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("district", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ], remainder="drop", sparse_threshold=0)


def models():
    """Bagging RF; boosted trees; stack trained with OOF predictions within train years."""
    rf = RandomForestClassifier(
        n_estimators=120, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    gb = HistGradientBoostingClassifier(
        max_iter=110, learning_rate=0.07, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42,
    )
    stack = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=80, max_depth=10, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )),
            ("boost", HistGradientBoostingClassifier(
                max_iter=80, max_leaf_nodes=13, learning_rate=0.08,
                l2_regularization=1.0, random_state=42,
            )),
            ("linear", LogisticRegression(max_iter=1200, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1200, random_state=42),
        cv=3, n_jobs=-1,
    )
    return {
        "Bagging — Random Forest": Pipeline([("prep", preprocess()), ("model", rf)]),
        "Boosting — HistGradientBoosting": Pipeline([("prep", preprocess()), ("model", gb)]),
        "Stacking — Meta Logistic": Pipeline([("prep", preprocess()), ("model", stack)]),
        "Baseline — Majority Class": Pipeline([
            ("prep", preprocess()),
            ("model", DummyClassifier(strategy="prior")),
        ]),
    }


def evaluate(model, df: pd.DataFrame) -> dict:
    x, y = df[FEATURES], df["rain_tomorrow"].astype(int)
    proba = model.predict_proba(x)[:, 1]
    labels = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, labels)), 4),
        "precision": round(float(precision_score(y, labels, zero_division=0)), 4),
        "recall": round(float(recall_score(y, labels, zero_division=0)), 4),
        "f1": round(float(f1_score(y, labels, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "tn_fp_fn_tp": [int(x) for x in confusion_matrix(y, labels, labels=[0, 1]).ravel()],
        "rain_prevalence": round(float(y.mean()), 4),
        "n_samples": int(len(y)),
    }


def run_experiment(train, validation, test, save=True, verbose=True):
    model_set = models()
    metrics = []
    prediction_frames = []
    for name, model in model_set.items():
        if verbose:
            print(f"Training: {name} ...", flush=True)
        model.fit(train[FEATURES], train["rain_tomorrow"].astype(int))
        for split_name, partition in [("validation_2024", validation), ("test_2025", test)]:
            scores = evaluate(model, partition)
            metrics.append({"model": name, "split": split_name, **scores})
            probs = model.predict_proba(partition[FEATURES])[:, 1]
            prediction_frames.append(pd.DataFrame({
                "date": partition["target_date"].to_numpy(),
                "district": partition["district"].to_numpy(),
                "actual": partition["rain_tomorrow"].to_numpy(),
                "model": name, "probability": probs,
                "prediction": (probs >= 0.5).astype(int),
                "split": split_name,
            }))
    metric_df = pd.DataFrame(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if save:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        for name, estimator in model_set.items():
            safe = name.split(" — ")[0].lower()
            joblib.dump(estimator, ARTIFACTS / f"{safe}.joblib", compress=3)
        metric_df.to_csv(ARTIFACTS / "metrics.csv", index=False)
        predictions.to_csv(ARTIFACTS / "predictions.csv", index=False)
        summary = {"dataset_rows": len(train) + len(validation) + len(test),
                   "train_rows": len(train), "validation_rows": len(validation),
                   "test_rows": len(test), "rain_threshold_mm": 1.0}
        (ARTIFACTS / "experiment.json").write_text(json.dumps(summary, indent=2))
    return model_set, metric_df, predictions


def read_models():
    return {
        "Bagging — Random Forest": joblib.load(ARTIFACTS / "bagging.joblib"),
        "Boosting — HistGradientBoosting": joblib.load(ARTIFACTS / "boosting.joblib"),
        "Stacking — Meta Logistic": joblib.load(ARTIFACTS / "stacking.joblib"),
        "Baseline — Majority Class": joblib.load(ARTIFACTS / "baseline.joblib"),
    }
