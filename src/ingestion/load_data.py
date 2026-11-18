
"""
Module: load_data.py
Purpose: Load and validate company datasets from CSV files.

Supports:
- Company A: ERP, invoice tool, taxonomy and labels.
- Company B: invoices, FX rates, taxonomy and seed labels.
- Company C: configurable input datasets.
"""

from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(file_path: str) -> Path:
    """
    Resolve a file path relative to the project root.

    Parameters
    ----------
    file_path : str
        Relative or absolute file path.

    Returns
    -------
    Path
        Resolved absolute path.
    """

    path = Path(file_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_csv(
    file_path: str,
    required_columns: Optional[List[str]] = None,
    dtype=None
) -> pd.DataFrame:
    """
    Load a CSV file and validate its required columns.

    Parameters
    ----------
    file_path : str
        CSV file location.
    required_columns : list, optional
        Columns that must exist in the dataset.
    dtype : optional
        Data types to apply while reading.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the file is empty or required columns are missing.
    """

    path = resolve_path(file_path)

    # Check file existence
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file but received: {path}"
        )

    try:
        # Read CSV
        df = pd.read_csv(
            path,
            low_memory=False,
            dtype=dtype,
            encoding="utf-8-sig"
        )

    except UnicodeDecodeError:
        # Fallback for files using Windows encoding
        df = pd.read_csv(
            path,
            low_memory=False,
            dtype=dtype,
            encoding="cp1252"
        )

    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"CSV file is empty: {path.name}"
        ) from exc

    # Validate required columns
    missing_columns = (
        set(required_columns or []) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in {path.name}: "
            f"{sorted(missing_columns)}"
        )

    # Remove accidental whitespace from column names
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        raise ValueError(
            f"Dataset contains no records: {path.name}"
        )

    print(
        f"[SUCCESS] Loaded {path.name} | "
        f"Rows: {len(df):,} | "
        f"Columns: {len(df.columns)}"
    )

    return df


def load_company_inputs(
    config: dict,
    required_columns: Optional[Dict[str, List[str]]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets defined in a company configuration.

    Parameters
    ----------
    config : dict
        Company configuration containing input file paths.
    required_columns : dict, optional
        Required columns for individual input datasets.

    Returns
    -------
    dict
        Dictionary containing loaded DataFrames.
    """

    if "inputs" not in config:
        raise ValueError(
            "Company configuration must contain an 'inputs' section."
        )

    datasets = {}

    for dataset_name, file_path in config["inputs"].items():

        if not file_path:
            continue

        columns = (required_columns or {}).get(
            dataset_name, []
        )

        datasets[dataset_name] = load_csv(
            file_path=file_path,
            required_columns=columns
        )

    print(
        f"\n[SUCCESS] Loaded {len(datasets)} "
        f"datasets for Company {config.get('company', 'Unknown')}."
    )

    return datasets
