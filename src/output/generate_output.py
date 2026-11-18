
"""
Generate, validate and save the final Cantill assignment submission CSV.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import OUTPUT_COLUMNS


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


def prepare_output(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data using the exact assignment output schema."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Output must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot generate an empty submission.")

    missing = [
        column for column in OUTPUT_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df[OUTPUT_COLUMNS].copy()

    # Normalize text columns.
    for column in TEXT_COLUMNS:
        result[column] = (
            result[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"nan": "", "None": "", "NaN": ""})
        )

    # Validate record IDs.
    if result["record_id"].eq("").any():
        raise ValueError("Output contains blank record IDs.")

    if result["record_id"].duplicated().any():
        duplicates = result.loc[
            result["record_id"].duplicated(keep=False),
            "record_id",
        ].unique().tolist()

        raise ValueError(
            f"Duplicate record IDs found: {duplicates[:10]}"
        )

    # Validate include.
    include = pd.to_numeric(
        result["include"],
        errors="coerce",
    )

    if include.isna().any() or not include.isin([0, 1]).all():
        raise ValueError("include must contain only 0 or 1.")

    result["include"] = include.astype(int)

    # Validate spend.
    result["spend"] = pd.to_numeric(
        result["spend"],
        errors="coerce",
    )

    if result["spend"].isna().any():
        raise ValueError("Spend contains missing or invalid values.")

    if not np.isfinite(result["spend"]).all():
        raise ValueError("Spend contains infinite values.")

    # Validate confidence.
    result["confidence"] = pd.to_numeric(
        result["confidence"],
        errors="coerce",
    )

    if result["confidence"].isna().any():
        raise ValueError(
            "Confidence contains missing or invalid values."
        )

    if not result["confidence"].between(0, 1).all():
        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    # Apply output business rules.
    result.loc[result["include"].eq(0), "spend"] = 0.0
    result.loc[result["include"].eq(1), "exclusion_reason"] = ""

    result["spend"] = result["spend"].round(2)
    result["confidence"] = result["confidence"].round(4)

    return result[OUTPUT_COLUMNS]


def validate_output(df: pd.DataFrame) -> bool:
    """Validate the final submission DataFrame."""

    prepared = prepare_output(df)

    if list(prepared.columns) != OUTPUT_COLUMNS:
        raise ValueError("Output schema does not match the assignment.")

    invalid_spend = (
        prepared["include"].eq(0)
        & prepared["spend"].ne(0)
    )

    if invalid_spend.any():
        raise ValueError("Excluded records must have zero spend.")

    invalid_reason = (
        prepared["include"].eq(1)
        & prepared["exclusion_reason"].ne("")
    )

    if invalid_reason.any():
        raise ValueError(
            "Included records must have a blank exclusion reason."
        )

    return True


def write_output(
    df: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Validate and save the final CSV atomically."""

    if path is None or not str(path).strip():
        raise ValueError("An output file path must be provided.")

    output_path = Path(path).expanduser().resolve()

    if output_path.suffix.lower() != ".csv":
        raise ValueError("Output path must end with .csv.")

    prepared = prepare_output(df)
    validate_output(prepared)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.stem + ".tmp" + output_path.suffix
    )

    try:
        prepared.to_csv(
            temporary_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

        os.replace(temporary_path, output_path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return output_path


def generate_output(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Convenience wrapper for write_output()."""

    return write_output(df, output_path)
