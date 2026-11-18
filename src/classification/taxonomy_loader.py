
"""Load and validate company-specific classification taxonomies."""

from pathlib import Path

import pandas as pd


def load_taxonomy(path: str | Path) -> pd.DataFrame:
    """
    Load a taxonomy CSV at runtime.

    Expected columns:
        category_l1, category_l2, category_l3

    category_l4 is optional.
    Additional non-category columns are preserved.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Taxonomy path is not a file: {path}")

    try:
        taxonomy = pd.read_csv(path, dtype="string")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Taxonomy file is empty: {path}") from exc

    if taxonomy.empty:
        raise ValueError(f"Taxonomy contains no records: {path}")

    # Normalize column names.
    taxonomy.columns = [
        str(column).strip().lower()
        for column in taxonomy.columns
    ]

    if taxonomy.columns.duplicated().any():
        duplicates = taxonomy.columns[
            taxonomy.columns.duplicated()
        ].tolist()
        raise ValueError(
            f"Duplicate column names found in taxonomy: {duplicates}"
        )

    # Identify category hierarchy columns.
    category_columns = [
        column
        for column in taxonomy.columns
        if column.startswith("category_l")
    ]

    if not category_columns:
        raise ValueError(
            "Taxonomy must contain category columns such as "
            "category_l1, category_l2 and category_l3."
        )

    # Require sequential levels starting from category_l1.
    expected_columns = [
        f"category_l{level}"
        for level in range(1, len(category_columns) + 1)
    ]

    missing = [
        column for column in expected_columns
        if column not in category_columns
    ]

    if missing:
        raise ValueError(
            f"Taxonomy hierarchy has missing levels: {missing}"
        )

    category_columns = expected_columns

    # Clean whitespace and missing values.
    taxonomy = taxonomy.fillna("")

    for column in taxonomy.columns:
        taxonomy[column] = (
            taxonomy[column]
            .astype("string")
            .str.strip()
        )

    # Remove rows with no category information.
    valid_rows = taxonomy[category_columns].ne("").any(axis=1)
    taxonomy = taxonomy.loc[valid_rows].copy()

    if taxonomy.empty:
        raise ValueError("Taxonomy has no valid category paths.")

    # Remove duplicate category paths.
    taxonomy = taxonomy.drop_duplicates(
        subset=category_columns,
        keep="first",
    )

    # Put category columns first and in hierarchy order.
    other_columns = [
        column
        for column in taxonomy.columns
        if column not in category_columns
    ]

    taxonomy = taxonomy[
        category_columns + other_columns
    ].reset_index(drop=True)

    return taxonomy


def get_taxonomy_levels(taxonomy: pd.DataFrame) -> list[str]:
    """Return category levels in hierarchy order."""

    levels = [
        column
        for column in taxonomy.columns
        if column.startswith("category_l")
    ]

    def level_number(column: str) -> int:
        return int(column.replace("category_l", ""))

    return sorted(levels, key=level_number)
