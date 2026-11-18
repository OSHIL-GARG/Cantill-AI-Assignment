
"""Evaluation metrics for purchase matching and duplicate detection."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _normalize_pairs(pairs) -> set[tuple[str, str]]:
    """
    Convert record pairs into a normalized set.

    Each pair should contain two record identifiers.
    Pair order is ignored because matching is treated as symmetric.
    """
    normalized = set()

    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(
                "Each match pair must contain exactly two record IDs."
            )

        left, right = pair

        if pd.isna(left) or pd.isna(right):
            continue

        left = str(left).strip()
        right = str(right).strip()

        if not left or not right or left == right:
            continue

        normalized.add(tuple(sorted((left, right))))

    return normalized


def evaluate_match_pairs(
    actual_matches,
    predicted_matches,
) -> dict[str, Any]:
    """
    Evaluate predicted record pairs against known correct pairs.

    Parameters
    ----------
    actual_matches:
        Iterable of correct (record_id_1, record_id_2) pairs.
    predicted_matches:
        Iterable of predicted (record_id_1, record_id_2) pairs.

    Returns
    -------
    Dictionary containing match precision, recall, F1 and counts.
    """
    actual = _normalize_pairs(actual_matches)
    predicted = _normalize_pairs(predicted_matches)

    true_positive = len(actual & predicted)
    false_positive = len(predicted - actual)
    false_negative = len(actual - predicted)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "actual_match_pairs": len(actual),
        "predicted_match_pairs": len(predicted),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "match_precision": float(precision),
        "match_recall": float(recall),
        "match_f1": float(f1),
    }


def matching_summary(
    total_records: int,
    matched_records: int,
    duplicate_records: int,
) -> dict[str, Any]:
    """
    Calculate a basic matching summary.

    This preserves the simple function interface used by the
    original project implementation.
    """
    total = int(total_records)
    matched = int(matched_records)
    duplicates = int(duplicate_records)

    if min(total, matched, duplicates) < 0:
        raise ValueError("Record counts cannot be negative.")

    if matched > total:
        raise ValueError("Matched records cannot exceed total records.")

    if duplicates > total:
        raise ValueError("Duplicate records cannot exceed total records.")

    return {
        "total_records": total,
        "matched_records": matched,
        "unmatched_records": total - matched,
        "duplicate_records": duplicates,
        "match_rate": float(matched / total) if total else 0.0,
    }


def duplicate_detection_metrics(
    actual_duplicate,
    predicted_duplicate,
) -> dict[str, Any]:
    """
    Evaluate duplicate detection using known duplicate labels.

    Values should be booleans or values convertible to booleans.
    """
    actual = pd.Series(actual_duplicate).fillna(False).astype(bool)
    predicted = pd.Series(predicted_duplicate).fillna(False).astype(bool)

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted duplicate labels must have equal lengths."
        )

    if actual.empty:
        return {
            "sample_count": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 0,
        }

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[False, True],
    ).ravel()

    return {
        "sample_count": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def matching_summary_from_dataframe(
    df: pd.DataFrame,
    matched_column: str = "is_matched",
    duplicate_column: str = "is_duplicate",
) -> dict[str, Any]:
    """
    Create a matching summary from a processed DataFrame.

    Missing matched/duplicate columns are treated as all False.
    """
    total = len(df)

    if matched_column in df.columns:
        matched = (
            df[matched_column]
            .fillna(False)
            .astype(bool)
            .sum()
        )
    else:
        matched = 0

    if duplicate_column in df.columns:
        duplicates = (
            df[duplicate_column]
            .fillna(False)
            .astype(bool)
            .sum()
        )
    else:
        duplicates = 0

    return matching_summary(
        total_records=total,
        matched_records=int(matched),
        duplicate_records=int(duplicates),
    )
