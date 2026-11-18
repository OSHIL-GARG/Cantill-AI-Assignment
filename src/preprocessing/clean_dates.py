
"""
Module: clean_dates.py

Purpose:
    Standardize date columns across company datasets.

Responsibilities:
    1. Parse different date formats.
    2. Handle missing dates.
    3. Detect invalid dates.
    4. Preserve original date values.
    5. Generate date quality reports.
"""

from typing import Optional, Dict, List

import pandas as pd


# --------------------------------------------------
# 1. Parse a Single Date Series
# --------------------------------------------------

def parse_date_series(
    series: pd.Series,
    dayfirst: bool = False,
    date_format: Optional[str] = None
) -> pd.Series:
    """
    Convert a date series into pandas datetime format.

    Parameters:
        series:
            Original date series.

        dayfirst:
            True for DD/MM/YYYY interpretation.
            False for MM/DD/YYYY interpretation.

        date_format:
            Optional explicit date format.

    Returns:
        Parsed datetime series.
    """

    if date_format:
        parsed = pd.to_datetime(
            series,
            format=date_format,
            errors="coerce"
        )
    else:
        # Mixed-format parsing supports different formats
        # within the same column.
        try:
            parsed = pd.to_datetime(
                series,
                format="mixed",
                errors="coerce",
                dayfirst=dayfirst
            )

        except (TypeError, ValueError):
            # Compatibility fallback for older pandas versions
            parsed = pd.to_datetime(
                series,
                errors="coerce",
                dayfirst=dayfirst
            )

    return parsed


# --------------------------------------------------
# 2. Clean a Single Date Column
# --------------------------------------------------

def clean_date_column(
    df: pd.DataFrame,
    column: str,
    dayfirst: bool = False,
    date_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Clean one date column.

    Original values are preserved in:
        <column>_original

    Parsed dates are stored in:
        <column>

    Quality flags:
        <column>_invalid
        <column>_missing
    """

    if column not in df.columns:
        raise ValueError(
            f"Date column not found: {column}"
        )

    result = df.copy()

    original_column = f"{column}_original"

    # Preserve original values
    result[original_column] = result[column]

    # Identify missing values before conversion
    original = result[column]

    missing_mask = (
        original.isna()
        | original.astype("string").str.strip().eq("")
    ).fillna(False)

    # Parse dates
    parsed_dates = parse_date_series(
        original,
        dayfirst=dayfirst,
        date_format=date_format
    )

    # Invalid means non-empty input that cannot be parsed
    invalid_mask = (
        ~missing_mask & parsed_dates.isna()
    )

    result[column] = parsed_dates

    result[f"{column}_invalid"] = invalid_mask

    result[f"{column}_missing"] = missing_mask

    return result


# --------------------------------------------------
# 3. Clean Multiple Date Columns
# --------------------------------------------------

def clean_date_columns(
    df: pd.DataFrame,
    date_columns: List[str],
    dayfirst: bool = False,
    date_formats: Optional[Dict[str, str]] = None,
    dayfirst_by_column: Optional[Dict[str, bool]] = None
) -> pd.DataFrame:
    """
    Clean multiple date columns.

    Parameters:
        df:
            Input DataFrame.

        date_columns:
            List of date columns.

        dayfirst:
            Default date interpretation.

        date_formats:
            Optional format mapping.

        dayfirst_by_column:
            Optional column-specific day-first settings.

    Returns:
        Cleaned DataFrame.
    """

    result = df.copy()

    date_formats = date_formats or {}

    dayfirst_by_column = dayfirst_by_column or {}

    for column in date_columns:

        column_dayfirst = dayfirst_by_column.get(
            column,
            dayfirst
        )

        result = clean_date_column(
            df=result,
            column=column,
            dayfirst=column_dayfirst,
            date_format=date_formats.get(column)
        )

    return result


# --------------------------------------------------
# 4. Generate Date Quality Report
# --------------------------------------------------

def generate_date_quality_report(
    df: pd.DataFrame,
    date_columns: List[str]
) -> pd.DataFrame:
    """
    Generate a quality report for date columns.
    """

    report = []

    for column in date_columns:

        if column not in df.columns:
            raise ValueError(
                f"Date column not found: {column}"
            )

        invalid_column = f"{column}_invalid"

        missing_column = f"{column}_missing"

        invalid_count = (
            int(df[invalid_column].sum())
            if invalid_column in df.columns
            else 0
        )

        missing_count = (
            int(df[missing_column].sum())
            if missing_column in df.columns
            else int(df[column].isna().sum())
        )

        total = len(df)

        report.append({
            "column": column,
            "total_records": total,
            "valid_dates": total - invalid_count - missing_count,
            "invalid_dates": invalid_count,
            "missing_dates": missing_count,
            "invalid_percentage": round(
                invalid_count / total * 100, 2
            ) if total else 0,
            "missing_percentage": round(
                missing_count / total * 100, 2
            ) if total else 0
        })

    return pd.DataFrame(report)


# --------------------------------------------------
# 5. Complete Date Cleaning Pipeline
# --------------------------------------------------

def clean_dates(
    df: pd.DataFrame,
    date_columns: List[str],
    dayfirst: bool = False,
    date_formats: Optional[Dict[str, str]] = None,
    dayfirst_by_column: Optional[Dict[str, bool]] = None
):
    """
    Execute the complete date cleaning pipeline.

    Returns:
        cleaned_df
        date_quality_report
    """

    print("\nStarting date cleaning...")

    cleaned_df = clean_date_columns(
        df=df,
        date_columns=date_columns,
        dayfirst=dayfirst,
        date_formats=date_formats,
        dayfirst_by_column=dayfirst_by_column
    )

    quality_report = generate_date_quality_report(
        cleaned_df,
        date_columns
    )

    print("\nDate cleaning completed.")

    print(quality_report.to_string(index=False))

    return cleaned_df, quality_report
