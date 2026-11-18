
"""Apply configurable rules to identify records excluded from spend."""

from __future__ import annotations

import re
import pandas as pd


DEFAULT_NON_PROCUREMENT_PATTERNS = [
    r"\binter[\s-]?company\b",
    r"\bintercompany\b",
    r"\broyalt(?:y|ies)\b",
    r"\bdividend\b",
    r"\bincome tax\b",
    r"\bvat payable\b",
    r"\bwithholding tax\b",
    r"\bcorporate tax\b",
    r"\brevenue\b",
    r"\bcustomer rebate\b",
    r"\bpartner billing\b",
    r"\binterest expense\b",
    r"\bloan repayment\b",
    r"\bfinancing cost\b",
]


def apply_exclusion_rules(
    df: pd.DataFrame,
    status_column: str | None = None,
    description_columns: list[str] | None = None,
    account_columns: list[str] | None = None,
    duplicate_column: str | None = None,
    non_procurement_patterns: list[str] | None = None,
    approved_statuses: tuple[str, ...] = ("approved", "paid"),
    cancelled_statuses: tuple[str, ...] = ("cancelled", "canceled"),
) -> pd.DataFrame:
    """
    Add include and exclusion_reason columns.

    The function does not delete records. Every input row is retained.

    A record is excluded when it is:
      - cancelled;
      - not approved or paid;
      - identified as non-procurement;
      - marked as a duplicate.

    Later rules do not overwrite an earlier exclusion reason.
    """
    result = df.copy()

    patterns = (
        non_procurement_patterns
        if non_procurement_patterns is not None
        else DEFAULT_NON_PROCUREMENT_PATTERNS
    )

    result["include"] = 1
    result["exclusion_reason"] = ""

    def exclude(mask: pd.Series, reason: str) -> None:
        """Apply a reason only to records not already excluded."""
        mask = mask.fillna(False).astype(bool)
        eligible = result["include"].eq(1)
        selected = mask & eligible

        result.loc[selected, "include"] = 0
        result.loc[selected, "exclusion_reason"] = reason

    # 1. Cancelled invoices or lines.
    if status_column and status_column in result.columns:
        status = (
            result[status_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        cancelled = status.isin(
            {value.lower() for value in cancelled_statuses}
        )
        exclude(cancelled, "CANCELLED")

        # 2. Only approved and paid records count.
        approved = status.isin(
            {value.lower() for value in approved_statuses}
        )

        # Blank statuses are not automatically excluded here.
        # This allows files without a status field to use other rules.
        has_status = status.ne("")
        exclude(has_status & ~approved & ~cancelled, "NOT_APPROVED")

    # 3. Non-procurement records.
    text_columns = (description_columns or []) + (account_columns or [])
    available_columns = [
        column for column in text_columns if column in result.columns
    ]

    if available_columns and patterns:
        combined_text = (
            result[available_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )

        pattern = "|".join(f"(?:{p})" for p in patterns)
        non_procurement = combined_text.str.contains(
            pattern,
            regex=True,
            na=False,
        )

        exclude(non_procurement, "NON_PROCUREMENT")

    # 4. Duplicate records.
    if duplicate_column and duplicate_column in result.columns:
        duplicate = (
            result[duplicate_column]
            .fillna(False)
            .astype(bool)
        )
        exclude(duplicate, "DUPLICATE")

    return result


def mark_exact_duplicates(
    df: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    """
    Mark repeated records using the supplied business-key columns.

    The first occurrence is retained. Later occurrences are marked
    as duplicates. Choose key columns carefully: do not use amount
    alone, because legitimate purchases can have the same amount.
    """
    missing = [c for c in key_columns if c not in df.columns]

    if missing:
        raise KeyError(f"Duplicate key columns not found: {missing}")

    result = df.copy()

    result["is_duplicate"] = result.duplicated(
        subset=key_columns,
        keep="first",
    )

    return result
