"""Train, evaluate and save a fake-news text classification model.

Example:
python -m src.train --fake-csv data/Fake.csv --true-csv data/True.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.text_processing import clean_text

RANDOM_STATE = 42
LABELS = ["fake", "real"]


def combine_title_and_text(frame: pd.DataFrame) -> pd.Series:
    """Return one document per record, accepting common Kaggle column names."""
    columns = {str(c).lower().strip(): c for c in frame.columns}
    title_col = columns.get("title")
    text_col = columns.get("text") or columns.get("content") or columns.get("article")
    if text_col is None and title_col is None:
        raise ValueError("Expected a title, text, content, or article column.")
    title = frame[title_col].fillna("").astype(str) if title_col else ""
    body = frame[text_col].fillna("").astype(str) if text_col else ""
    return (title + " " + body).str.strip()


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    """Load either one labelled CSV or the usual Fake.csv / True.csv pair."""
    if args.data:
        data = pd.read_csv(args.data)
        cols = {str(c).lower().strip(): c for c in data.columns}
        label_col = cols.get("label")
        target_col = cols.get("target")
        if label_col is None and target_col is None:
            raise ValueError("The --data file must have a 'label' column or a supported 'target' column.")
        if label_col is None:
            result_label = data[target_col].replace({"0": "real", "1": "fake", 0: "real", 1: "fake"})
        else:
            result_label = data[label_col]
        result = pd.DataFrame({"text": combine_title_and_text(data), "label": result_label})
    else:
        if not (args.fake_csv and args.true_csv):
            raise ValueError("Provide --data OR both --fake-csv and --true-csv.")
        fake = pd.read_csv(args.fake_csv)
        real = pd.read_csv(args.true_csv)
        result = pd.concat([
            pd.DataFrame({"text": combine_title_and_text(fake), "label": "fake"}),
            pd.DataFrame({"text": combine_title_and_text(real), "label": "real"}),
        ], ignore_index=True)

    result = result.dropna(subset=["text", "label"]).copy()
    result["text"] = result["text"].astype(str).str.strip()
    result = result[result["text"].str.len() >= args.min_characters]
    result["label"] = result["label"].astype(str).str.strip().str.lower()
    result["label"] = result["label"].replace({"0": "fake", "1": "real", "false": "fake", "true": "real"})
    unknown = set(result["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Labels must be fake/real or 0/1; found {sorted(unknown)}")
    if result["label"].nunique() != 2:
        raise ValueError("Training needs examples for both fake and real labels.")
    return result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def split_data(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create reproducible 70/15/15 train, validation and final test sets."""
    train, holdout = train_test_split(data, test_size=0.30, stratify=data.label, random_state=RANDOM_STATE)
    validation, test = train_test_split(holdout, test_size=0.50, stratify=holdout.label, random_state=RANDOM_STATE)
    return train.text, train.label, validation.text, validation.label, test.text, test.label


def make_pipeline(max_features: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_df=0.95, max_features=max_features, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def metrics_for(y_true: pd.Series, predictions) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, labels=LABELS, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "fake": {"precision": round(float(precision[0]), 4), "recall": round(float(recall[0]), 4), "f1": round(float(f1[0]), 4)},
        "real": {"precision": round(float(precision[1]), 4), "recall": round(float(recall[1]), 4), "f1": round(float(f1[1]), 4)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fake news NLP model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="CSV with text/title and label columns")
    source.add_argument("--fake-csv", type=Path, help="CSV containing fake articles")
    parser.add_argument("--true-csv", type=Path, help="CSV containing genuine articles (required with --fake-csv)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--min-characters", type=int, default=40)
    parser.add_argument("--max-features", type=int, default=100000)
    args = parser.parse_args()

    data = load_data(args)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(data)
    model = make_pipeline(args.max_features)
    model.fit(x_train, y_train)
    validation_metrics = metrics_for(y_val, model.predict(x_val))
    test_predictions = model.predict(x_test)
    test_metrics = metrics_for(y_test, test_predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "fake_news_model.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "dataset_rows": int(len(data)), "splits": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "validation": validation_metrics, "test": test_metrics,
        "classification_report": classification_report(y_test, test_predictions, labels=LABELS, zero_division=0, output_dict=True),
    }, indent=2), encoding="utf-8")
    pd.DataFrame({"text": x_test, "actual": y_test, "predicted": test_predictions}).to_csv(args.output_dir / "test_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, test_predictions, labels=LABELS), display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Fake News Detection: Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    print(json.dumps({"validation": validation_metrics, "test": test_metrics}, indent=2))
    print(f"Saved model and reports to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
