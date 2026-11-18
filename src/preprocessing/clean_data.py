
"""
Module: clean_data.py

Purpose:
    Perform general data cleaning and standardization
    for the Cantill Spend Intelligence project.

Responsibilities:
    1. Standardize column names.
    2. Clean string values.
    3. Handle missing values.
    4. Normalize text.
    5. Identify duplicate records.
    6. Generate data quality reports.

Important:
    Original records are never automatically deleted.
"""

import re
import unicodedata
from typing import Optional, List

import pandas as pd


# --------------------------------------------------
# 1. Standardize Column Names
# --------------------------------------------------

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names into a consistent format.

    Example:
        'Invoice Date' -> 'invoice_date'
        'Supplier Name' -> 'supplier_name'
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cleaned_df = df.copy()

    cleaned_columns = []

    for column in cleaned_df.columns:

        name = str(column).strip().lower()

        # Replace spaces and special characters with underscores
        name = re.sub(r"[^a-z0-9]+", "_", name)

        name = name.strip("_")

        cleaned_columns.append(name)

    # Prevent accidental duplicate column names
    if len(cleaned_columns) != len(set(cleaned_columns)):
        raise ValueError(
            "Duplicate column names found after standardization."
        )

    cleaned_df.columns = cleaned_columns

    return cleaned_df


# --------------------------------------------------
# 2. Clean Text Values
# --------------------------------------------------

def clean_text_column(series: pd.Series) -> pd.Series:
    """
    Clean text values without removing meaningful information.

    Operations:
        - Remove leading/trailing whitespace.
        - Normalize repeated spaces.
        - Convert empty strings into missing values.
        - Preserve multilingual characters.
    """

    def clean_value(value):

        if pd.isna(value):
            return pd.NA

        text = str(value)

        # Normalize Unicode representation
        text = unicodedata.normalize("NFC", text)

        # Remove leading and trailing whitespace
        text = text.strip()

        # Replace multiple whitespace characters
        text = re.sub(r"\s+", " ", text)

        if not text:
            return pd.NA

        # Handle common missing-value strings
        if text.lower() in {
            "nan",
            "none",
            "null",
            "n/a",
            "na",
            "missing"
        }:
            return pd.NA

        return text

    return series.map(clean_value)


# --------------------------------------------------
# 3. Normalize Text For Matching
# --------------------------------------------------

def normalize_text_for_matching(series: pd.Series) -> pd.Series:
    """
    Create normalized text suitable for matching suppliers,
    items and purchase descriptions.

    Original text should be preserved separately.

    Example:
        ' ACME  INDUSTRIES Ltd. '
        becomes:
        'acme industries ltd'
    """

    def normalize(value):

        if pd.isna(value):
            return ""

        text = unicodedata.normalize("NFKC", str(value))

        text = text.lower().strip()

        # Preserve Unicode letters and numbers
        text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    return series.map(normalize)


# --------------------------------------------------
# 4. Clean Entire DataFrame
# --------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    create_normalized_columns: bool = False
) -> pd.DataFrame:
    """
    Perform general cleaning on an entire DataFrame.

    Parameters:
        df:
            Input DataFrame.

        text_columns:
            Columns containing textual information.

        create_normalized_columns:
            Whether to create additional normalized columns.

    Returns:
        Cleaned DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot clean an empty DataFrame.")

    cleaned_df = standardize_column_names(df)

    # Identify text columns automatically if not provided
    if text_columns is None:

        text_columns = cleaned_df.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

    # Standardize supplied column names
    text_columns = [
        str(col).strip().lower()
        for col in text_columns
    ]

    missing_columns = set(text_columns) - set(cleaned_df.columns)

    if missing_columns:
        raise ValueError(
            f"Text columns not found: {sorted(missing_columns)}"
        )

    for column in text_columns:

        # Clean original text
        cleaned_df[column] = clean_text_column(
            cleaned_df[column]
        )

        # Create a separate normalized version for matching
        if create_normalized_columns:

            normalized_column = f"{column}_normalized"

            cleaned_df[normalized_column] = (
                normalize_text_for_matching(
                    cleaned_df[column]
                )
            )

    return cleaned_df


# --------------------------------------------------
# 5. Detect Duplicate Records
# --------------------------------------------------

def detect_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Identify duplicate records without deleting them.

    Parameters:
        df:
            Input DataFrame.

        subset:
            Columns used for duplicate detection.

    Returns:
        DataFrame containing duplicate flags.
    """

    result = df.copy()

    duplicate_flags = result.duplicated(
        subset=subset,
        keep="first"
    )

    result["is_duplicate"] = duplicate_flags

    return result


# --------------------------------------------------
# 6. Generate Data Quality Report
# --------------------------------------------------

def generate_data_quality_report(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate a data quality report containing:

        - Column names
        - Data types
        - Missing values
        - Missing percentages
        - Unique values
        - Duplicate count
    """

    total_rows = len(df)

    report = pd.DataFrame({
        "column_name": df.columns,
        "data_type": [
            str(df[column].dtype)
            for column in df.columns
        ],
        "missing_values": [
            df[column].isna().sum()
            for column in df.columns
        ],
        "missing_percentage": [
            round(
                df[column].isna().mean() * 100, 2
            )
            for column in df.columns
        ],
        "unique_values": [
            df[column].nunique(dropna=True)
            for column in df.columns
        ]
    })

    report.attrs["total_rows"] = total_rows
    report.attrs["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    return report


# --------------------------------------------------
# 7. Complete Cleaning Pipeline
# --------------------------------------------------

def clean_data(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    create_normalized_columns: bool = False
):
    """
    Execute the general data cleaning pipeline.

    Returns:
        cleaned_df:
            Cleaned dataset.

        quality_report:
            Data quality summary.
    """

    print("\nStarting data cleaning...")

    original_rows = len(df)

    # Step 1: Clean DataFrame
    cleaned_df = clean_dataframe(
        df=df,
        text_columns=text_columns,
        create_normalized_columns=create_normalized_columns
    )

    # Step 2: Detect duplicates
    cleaned_df = detect_duplicates(cleaned_df)

    # Step 3: Generate quality report
    quality_report = generate_data_quality_report(
        cleaned_df
    )

    print("\nData cleaning completed.")

    print(f"Original records: {original_rows:,}")

    print(
        f"Duplicate records detected: "
        f"{cleaned_df['is_duplicate'].sum():,}"
    )

    print(
        f"Total columns after cleaning: "
        f"{len(cleaned_df.columns)}"
    )

    return cleaned_df, quality_report
