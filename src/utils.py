
"""Shared utilities for the Cantill Spend Intelligence project."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "record_id",
    "include",
    "exclusion_reason",
    "supplier_key",
    "item_key",
    "spend",
    "l1",
    "l2",
    "l3",
    "l4",
    "confidence",
]

TEXT_COLUMNS = [
    "record_id",
    "exclusion_reason",
    "supplier_key",
    "item_key",
    "l1",
    "l2",
    "l3",
    "l4",
]


def is_missing(value) -> bool:
    """Return True for None, NaN, or pandas missing values."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def norm_text(value) -> str:
    """Normalize text for matching and key generation."""
    if is_missing(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_id(value) -> str:
    """Normalize identifiers by retaining only uppercase letters and digits."""
    if is_missing(value):
        return ""

    text = str(value).strip().upper()

    if text in {"", "NAN", "NONE", "NULL"}:
        return ""

    return re.sub(r"[^A-Z0-9]", "", text)


def parse_amount(series: pd.Series) -> pd.Series:
    """
    Convert amount values into numeric values.

    Supports:
    - 1,250.50
    - 1250.50
    - (1250.50) as a negative amount
    - Currency symbols and surrounding spaces
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    values = series.astype("string").str.strip()

    negative_parentheses = values.str.match(r"^\(.*\)$", na=False)

    values = values.str.replace(",", "", regex=False)
    values = values.str.replace(r"[^\d.\-]", "", regex=True)

    numbers = pd.to_numeric(values, errors="coerce")

    numbers.loc[negative_parentheses] = (
        numbers.loc[negative_parentheses].abs() * -1
    )

    return numbers.fillna(0.0).astype(float)


def parse_dates(series: pd.Series) -> pd.Series:
    """Convert a date series to pandas datetime values."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    return pd.to_datetime(
        series,
        errors="coerce",
        dayfirst=False,
        format="mixed",
    )


def stable_key(prefix: str, value) -> str:
    """
    Generate a repeatable key from a normalized value.

    Example:
        stable_key("SUP", "ABC Suppliers Ltd")
        -> SUP-<12-character hash>
    """
    normalized = norm_text(value)

    if not normalized:
        return ""

    digest = hashlib.sha1(
        normalized.encode("utf-8")
    ).hexdigest()[:12]

    return f"{prefix}-{digest}"


def first_non_empty(*values) -> str:
    """Return the first non-missing, non-empty value."""
    for value in values:
        if not is_missing(value):
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none", "null"}:
                return text
    return ""


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    source_name: str,
) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}"
        )


def blank_output(record_ids) -> pd.DataFrame:
    """Create a correctly structured output for excluded records."""
    output = pd.DataFrame({
        "record_id": list(record_ids)
    })

    output["include"] = 0
    output["exclusion_reason"] = "NON_PROCUREMENT"

    for column in [
        "supplier_key",
        "item_key",
        "l1",
        "l2",
        "l3",
        "l4",
    ]:
        output[column] = ""

    output["spend"] = 0.0
    output["confidence"] = 0.0

    return output[OUTPUT_COLUMNS]


def validate_output_schema(dataframe: pd.DataFrame) -> None:
    """Validate required output columns and basic values."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Output must be a pandas DataFrame.")

    missing = [
        column
        for column in OUTPUT_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(f"Missing output columns: {missing}")

    if dataframe.empty:
        raise ValueError("Output DataFrame is empty.")

    if dataframe["record_id"].isna().any():
        raise ValueError("Output contains missing record IDs.")

    if dataframe["record_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Output contains blank record IDs.")

    include = pd.to_numeric(
        dataframe["include"], errors="coerce"
    )

    if include.isna().any() or not include.isin([0, 1]).all():
        raise ValueError("include must contain only 0 or 1.")

    spend = pd.to_numeric(
        dataframe["spend"], errors="coerce"
    )

    if spend.isna().any() or not np.isfinite(spend).all():
        raise ValueError("spend contains invalid numeric values.")

    confidence = pd.to_numeric(
        dataframe["confidence"], errors="coerce"
    )

    if (
        confidence.isna().any()
        or not confidence.between(0, 1).all()
    ):
        raise ValueError("confidence must be between 0 and 1.")


def prepare_output(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a clean DataFrame using the exact assignment schema."""
    validate_output_schema(dataframe)

    output = dataframe[OUTPUT_COLUMNS].copy()

    output["include"] = pd.to_numeric(
        output["include"], errors="raise"
    ).astype(int)

    output["spend"] = pd.to_numeric(
        output["spend"], errors="raise"
    ).round(2)

    output["confidence"] = pd.to_numeric(
        output["confidence"], errors="raise"
    ).clip(0, 1).round(4)

    for column in TEXT_COLUMNS:
        output[column] = (
            output[column]
            .fillna("")
            .astype(str)
            .replace({"nan": "", "None": ""})
        )

    output.loc[
        output["include"].eq(0), "spend"
    ] = 0.0

    output.loc[
        output["include"].eq(1), "exclusion_reason"
    ] = ""

    return output[OUTPUT_COLUMNS]
