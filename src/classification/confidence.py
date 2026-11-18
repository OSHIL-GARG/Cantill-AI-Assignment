
"""Validate and interpret classification confidence scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_confidence(values) -> list[float]:
    """
    Convert confidence values to floats between 0 and 1.

    Invalid, missing and infinite values become 0.0.
    """

    if np.isscalar(values):
        values = [values]

    series = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    series = series.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    series = series.clip(lower=0.0, upper=1.0)

    return series.astype(float).tolist()


def confidence_level(
    confidence: float,
    high_threshold: float = 0.75,
    medium_threshold: float = 0.45,
) -> str:
    """
    Assign a confidence label.

    HIGH:   score >= 0.75
    MEDIUM: score >= 0.45 and < 0.75
    LOW:    score < 0.45
    """

    if not (
        0 <= medium_threshold <= high_threshold <= 1
    ):
        raise ValueError(
            "Thresholds must satisfy "
            "0 <= medium_threshold <= high_threshold <= 1."
        )

    normalized = normalize_confidence([confidence])[0]

    if normalized >= high_threshold:
        return "HIGH"

    if normalized >= medium_threshold:
        return "MEDIUM"

    return "LOW"


def add_confidence_levels(
    df: pd.DataFrame,
    confidence_column: str = "confidence",
    high_threshold: float = 0.75,
    medium_threshold: float = 0.45,
) -> pd.DataFrame:
    """Add normalized confidence scores and confidence labels."""

    if confidence_column not in df.columns:
        raise KeyError(
            f"Confidence column not found: {confidence_column}"
        )

    # Validate thresholds before processing.
    if not (
        0 <= medium_threshold <= high_threshold <= 1
    ):
        raise ValueError(
            "Thresholds must satisfy "
            "0 <= medium_threshold <= high_threshold <= 1."
        )

    result = df.copy()

    result[confidence_column] = normalize_confidence(
        result[confidence_column]
    )

    result["confidence_level"] = result[confidence_column].apply(
        lambda score: confidence_level(
            score,
            high_threshold=high_threshold,
            medium_threshold=medium_threshold,
        )
    )

    return result
