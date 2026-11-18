
"""Clean and prepare text for purchase category classification."""

import pandas as pd

from ..utils import norm_text


def clean_classification_text(value) -> str:
    """Normalize a text value using the project's shared text utility."""

    return norm_text(value)


def prepare_classification_text(
    description,
    supplier="",
    material="",
    extra_text=None,
) -> str:
    """
    Combine description, supplier and material information.

    Empty values are ignored.
    """

    values = [
        description,
        supplier,
        material,
    ]

    if extra_text is not None:
        if isinstance(extra_text, (list, tuple)):
            values.extend(extra_text)
        else:
            values.append(extra_text)

    cleaned_values = []

    for value in values:
        cleaned = clean_classification_text(value)

        if cleaned:
            cleaned_values.append(cleaned)

    return " ".join(cleaned_values)


def prepare_text_from_dataframe(
    df: pd.DataFrame,
    description_column: str,
    supplier_column: str | None = None,
    material_column: str | None = None,
    extra_columns: list[str] | None = None,
    output_column: str = "classification_text",
) -> pd.DataFrame:
    """
    Generate a classification text column from DataFrame columns.

    Required:
        description_column

    Optional:
        supplier_column
        material_column
        extra_columns
    """

    if description_column not in df.columns:
        raise KeyError(
            f"Description column not found: {description_column}"
        )

    requested_columns = [
        supplier_column,
        material_column,
        *(extra_columns or []),
    ]

    missing_columns = [
        column
        for column in requested_columns
        if column and column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Optional text columns not found: {missing_columns}"
        )

    result = df.copy()

    def build_text(row):
        supplier = (
            row[supplier_column]
            if supplier_column else ""
        )

        material = (
            row[material_column]
            if material_column else ""
        )

        additional = [
            row[column]
            for column in (extra_columns or [])
        ]

        return prepare_classification_text(
            description=row[description_column],
            supplier=supplier,
            material=material,
            extra_text=additional,
        )

    result[output_column] = result.apply(build_text, axis=1)

    return result
