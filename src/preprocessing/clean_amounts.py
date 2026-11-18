
"""
Module: clean_amounts.py

Purpose:
    Clean and standardize financial amount columns.

Responsibilities:
    1. Parse numeric amounts from strings.
    2. Handle currency symbols and separators.
    3. Handle negative amounts.
    4. Normalize VAT rates.
    5. Calculate net amounts from gross amounts.
    6. Generate amount quality reports.

Important:
    Currency conversion is handled separately.
    This module does not convert currencies.
"""

import re
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Parse a Single Amount
# --------------------------------------------------

def parse_amount_value(value):
    """
    Convert an individual amount into a numeric value.

    Supported examples:
        '1,250.50'  -> 1250.50
        '1250.50'   -> 1250.50
        '(500.00)'  -> -500.00
        '-250'      -> -250.00
        'R 1,250'   -> 1250.00
        '$500'      -> 500.00

    Ambiguous separators are not silently guessed.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if np.isfinite(value) else np.nan

    text = str(value).strip()

    if not text or text.lower() in {
        "nan", "none", "null", "n/a", "na", "missing"
    }:
        return np.nan

    # Detect accounting-style negative values
    negative = text.startswith("(") and text.endswith(")")

    if negative:
        text = text[1:-1].strip()

    # Remove common currency symbols and codes.
    text = re.sub(
        r"(?i)\b(?:USD|ZAR|EUR|GBP|INR|AUD|CAD|JPY)\b",
        "",
        text
    )

    text = re.sub(r"[$€£₹]", "", text)
    text = re.sub(r"(?i)\bR\b", "", text)

    # Remove whitespace
    text = text.replace(" ", "")

    # Preserve digits, separators and sign
    if not re.fullmatch(r"[+-]?[\d.,]+", text):
        return np.nan

    # Handle both decimal and thousands separators
    if "," in text and "." in text:

        # Whichever separator appears last is treated
        # as the decimal separator.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "")
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

    elif "," in text:

        parts = text.split(",")

        # Example: 1,234 or 12,345,678
        if len(parts) > 2:
            if all(len(part) == 3 for part in parts[1:]):
                text = "".join(parts)
            else:
                return np.nan

        elif len(parts) == 2:

            # Example: 1,234 -> thousands separator
            if len(parts[1]) == 3 and len(parts[0]) > 0:
                text = "".join(parts)

            # Example: 1250,50 -> decimal separator
            else:
                text = ".".join(parts)

    elif text.count(".") > 1:

        parts = text.split(".")

        if all(len(part) == 3 for part in parts[1:]):
            text = "".join(parts)
        else:
            return np.nan

    try:
        amount = float(text)

        if not np.isfinite(amount):
            return np.nan

        return -abs(amount) if negative else amount

    except (ValueError, OverflowError):
        return np.nan


# --------------------------------------------------
# 2. Parse an Amount Series
# --------------------------------------------------

def parse_amount_series(series: pd.Series) -> pd.Series:
    """
    Convert a complete amount column into numeric values.
    """

    return pd.to_numeric(
        series.map(parse_amount_value),
        errors="coerce"
    ).astype("float64")


# --------------------------------------------------
# 3. Clean Amount Column
# --------------------------------------------------

def clean_amount_column(
    df: pd.DataFrame,
    column: str,
    output_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Clean an individual amount column.

    Original values are preserved.

    Parameters:
        df:
            Input DataFrame.

        column:
            Source amount column.

        output_column:
            Destination numeric column.

    Creates:
        amount column
        <output_column>_invalid
        <output_column>_missing
    """

    if column not in df.columns:
        raise ValueError(
            f"Amount column not found: {column}"
        )

    result = df.copy()

    output_column = output_column or f"{column}_numeric"

    original = result[column]

    missing_mask = (
        original.isna()
        | original.astype("string").str.strip().eq("")
    ).fillna(False)

    numeric_values = parse_amount_series(original)

    invalid_mask = (
        ~missing_mask & numeric_values.isna()
    )

    result[output_column] = numeric_values

    result[f"{output_column}_invalid"] = invalid_mask

    result[f"{output_column}_missing"] = missing_mask

    return result


# --------------------------------------------------
# 4. Normalize VAT Rates
# --------------------------------------------------

def normalize_vat_rate(
    vat_series: pd.Series,
    default_rate: float = 0.15
) -> pd.Series:
    """
    Convert VAT rates into decimal representation.

    Examples:
        15     -> 0.15
        15%    -> 0.15
        0.15   -> 0.15
        0      -> 0.00
        blank  -> default rate

    Rates must be supplied as percentages or decimals.
    """

    def convert_rate(value):

        if pd.isna(value):
            return default_rate

        text = str(value).strip()

        if not text:
            return default_rate

        percentage = text.endswith("%")

        if percentage:
            text = text[:-1].strip()

        try:
            rate = float(text)

            if not np.isfinite(rate) or rate < 0:
                return np.nan

            # Explicit percentages always divide by 100.
            if percentage or rate > 1:
                rate = rate / 100

            if rate > 1:
                return np.nan

            return rate

        except (ValueError, TypeError):
            return np.nan

    return vat_series.map(convert_rate).astype("float64")


# --------------------------------------------------
# 5. Calculate Net Amount From Gross Amount
# --------------------------------------------------

def calculate_net_amount(
    gross_amount: pd.Series,
    vat_rate: pd.Series,
    default_vat: float = 0.15
) -> pd.Series:
    """
    Remove VAT from gross amounts.

    Formula:
        Net Amount = Gross Amount / (1 + VAT Rate)

    VAT rate must be in decimal representation.
    """

    gross = parse_amount_series(gross_amount)

    vat = normalize_vat_rate(
        vat_rate,
        default_rate=default_vat
    )

    return gross / (1 + vat)


# --------------------------------------------------
# 6. Prepare Net Amount Column
# --------------------------------------------------

def add_net_amount_column(
    df: pd.DataFrame,
    gross_column: str,
    vat_column: str,
    output_column: str = "net_amount",
    default_vat: float = 0.15
) -> pd.DataFrame:
    """
    Add a net amount column to the DataFrame.

    Useful for Company A invoice-tool records.
    """

    required = [gross_column, vat_column]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns not found: {missing}"
        )

    result = df.copy()

    result[f"{gross_column}_numeric"] = (
        parse_amount_series(result[gross_column])
    )

    result["vat_rate_normalized"] = normalize_vat_rate(
        result[vat_column],
        default_rate=default_vat
    )

    result[output_column] = calculate_net_amount(
        result[gross_column],
        result[vat_column],
        default_vat=default_vat
    )

    return result


# --------------------------------------------------
# 7. Generate Amount Quality Report
# --------------------------------------------------

def generate_amount_quality_report(
    df: pd.DataFrame,
    amount_columns: List[str]
) -> pd.DataFrame:
    """
    Generate a report for amount columns.
    """

    report = []

    for column in amount_columns:

        if column not in df.columns:
            raise ValueError(
                f"Amount column not found: {column}"
            )

        values = parse_amount_series(df[column])

        valid_values = values.dropna()

        report.append({
            "column": column,
            "total_records": len(df),
            "valid_amounts": int(values.notna().sum()),
            "missing_or_invalid": int(values.isna().sum()),
            "negative_amounts": int((values < 0).sum()),
            "zero_amounts": int((values == 0).sum()),
            "positive_amounts": int((values > 0).sum()),
            "minimum": (
                float(valid_values.min())
                if not valid_values.empty else np.nan
            ),
            "maximum": (
                float(valid_values.max())
                if not valid_values.empty else np.nan
            )
        })

    return pd.DataFrame(report)


# --------------------------------------------------
# 8. Complete Amount Cleaning Pipeline
# --------------------------------------------------

def clean_amounts(
    df: pd.DataFrame,
    amount_columns: List[str]
):
    """
    Clean multiple amount columns and generate a report.

    Returns:
        cleaned_df
        amount_quality_report
    """

    print("\nStarting amount cleaning...")

    cleaned_df = df.copy()

    for column in amount_columns:

        cleaned_df = clean_amount_column(
            cleaned_df,
            column
        )

    report = generate_amount_quality_report(
        cleaned_df,
        amount_columns
    )

    print("\nAmount cleaning completed.")

    print(report.to_string(index=False))

    return cleaned_df, report
