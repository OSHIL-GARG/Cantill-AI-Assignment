
"""Evaluation metrics for hierarchical purchase classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _clean_labels(values) -> list[str]:
    """Convert labels to strings and represent missing labels consistently."""
    cleaned = []

    for value in values:
        if pd.isna(value):
            cleaned.append("")
        else:
            cleaned.append(str(value).strip())

    return cleaned


def classification_metrics(
    y_true,
    y_pred,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Calculate standard classification metrics.

    Parameters
    ----------
    y_true:
        Actual category labels.
    y_pred:
        Predicted category labels.
    labels:
        Optional complete set of category labels. Useful when some
        categories do not appear in the evaluation sample.

    Returns
    -------
    Dictionary containing accuracy, macro/weighted precision,
    recall and F1, plus the classification report and confusion matrix.
    """
    actual = _clean_labels(y_true)
    predicted = _clean_labels(y_pred)

    if len(actual) != len(predicted):
        raise ValueError(
            "y_true and y_pred must have the same number of records."
        )

    if not actual:
        return {
            "sample_count": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "classification_report": {},
            "confusion_matrix": [],
            "labels": [],
        }

    evaluation_labels = (
        sorted(set(actual) | set(predicted))
        if labels is None
        else sorted(set(_clean_labels(labels)) | set(actual) | set(predicted))
    )

    return {
        "sample_count": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(
            precision_score(
                actual,
                predicted,
                labels=evaluation_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                actual,
                predicted,
                labels=evaluation_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=evaluation_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                actual,
                predicted,
                labels=evaluation_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                actual,
                predicted,
                labels=evaluation_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted,
                labels=evaluation_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=evaluation_labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=evaluation_labels,
        ).tolist(),
        "labels": evaluation_labels,
    }


def hierarchical_classification_metrics(
    actual_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    levels: tuple[str, ...] = (
        "category_l1",
        "category_l2",
        "category_l3",
        "category_l4",
    ),
) -> dict[str, Any]:
    """
    Evaluate classification separately at each hierarchy level.

    Both DataFrames must contain the same number of rows in the
    same order. Rows with missing actual labels at a level are
    ignored for that level.
    """
    if len(actual_df) != len(predicted_df):
        raise ValueError(
            "Actual and predicted DataFrames must have equal row counts."
        )

    results = {}

    for level in levels:
        if level not in actual_df.columns:
            continue

        if level not in predicted_df.columns:
            continue

        actual = _clean_labels(actual_df[level])
        predicted = _clean_labels(predicted_df[level])

        # A blank actual category is not a labelled evaluation example.
        valid_positions = [
            index
            for index, value in enumerate(actual)
            if value != ""
        ]

        filtered_actual = [actual[i] for i in valid_positions]
        filtered_predicted = [predicted[i] for i in valid_positions]

        results[level] = classification_metrics(
            filtered_actual,
            filtered_predicted,
        )

    return results


def evaluate_classification_dataframe(
    df: pd.DataFrame,
    actual_prefix: str = "actual_",
    predicted_prefix: str = "predicted_",
    levels: tuple[str, ...] = (
        "category_l1",
        "category_l2",
        "category_l3",
        "category_l4",
    ),
) -> dict[str, Any]:
    """
    Evaluate a single DataFrame containing actual and predicted labels.

    Example columns:
        actual_category_l1
        predicted_category_l1
        actual_category_l2
        predicted_category_l2
    """
    results = {}

    for level in levels:
        actual_column = f"{actual_prefix}{level}"
        predicted_column = f"{predicted_prefix}{level}"

        if (
            actual_column not in df.columns
            or predicted_column not in df.columns
        ):
            continue

        actual = df[actual_column]
        predicted = df[predicted_column]

        valid = actual.notna() & actual.astype(str).str.strip().ne("")

        results[level] = classification_metrics(
            actual[valid],
            predicted[valid],
        )

    return results


def confidence_summary(
    df: pd.DataFrame,
    confidence_column: str = "confidence",
) -> dict[str, float | int]:
    """
    Summarize prediction confidence.

    These are descriptive statistics, not classification accuracy.
    """
    if confidence_column not in df.columns:
        raise KeyError(
            f"Confidence column not found: {confidence_column}"
        )

    values = pd.to_numeric(
        df[confidence_column],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    values = values.clip(0, 1)

    if values.empty:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }

    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }
