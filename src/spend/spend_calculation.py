
"""Calculate net spend by removing tax where required."""

from __future__ import annotations

import pandas as pd


def calculate_net_spend(
    df: pd.DataFrame,
    amount_column: str,
    amount_type: str = "net",
    vat_column: str | None = None,
    default_vat_rate: float = 0.15,
) -> pd.DataFrame:
    """
    Add raw_amount, vat_rate_used and net_spend columns.

    Parameters
    ----------
    df:
        Input purchase records.
    amount_column:
        Column containing the purchase amount.
    amount_type:
        'net' for amounts excluding VAT, or 'gross' for amounts
        including VAT.
    vat_column:
        Optional column containing VAT rates. Rates may be stored
        as 0.15 or 15.
    default_vat_rate:
        VAT rate used when the supplied rate is missing.
        Default is 15%.

    Returns
    -------
    DataFrame:
        Copy of the input with calculated spend columns.
    """
    if amount_column not in df.columns:
        raise KeyError(f"Amount column not found: {amount_column}")

    amount_type = amount_type.lower().strip()

    if amount_type not in {"net", "gross"}:
        raise ValueError("amount_type must be either 'net' or 'gross'.")

    if not 0 <= default_vat_rate <= 1:
        raise ValueError("default_vat_rate must be between 0 and 1.")

    result = df.copy()

    amount = pd.to_numeric(
        result[amount_column]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False),
        errors="coerce",
    )

    result["raw_amount"] = amount

    if amount_type == "net":
        result["vat_rate_used"] = 0.0
        result["net_spend"] = amount

    else:
        if vat_column and vat_column in result.columns:
            vat = pd.to_numeric(
                result[vat_column]
                .astype("string")
                .str.replace("%", "", regex=False),
                errors="coerce",
            )

            # Convert values such as 15 into 0.15.
            vat = vat.where(vat <= 1, vat / 100)

            # Invalid or missing rates use the configured default.
            vat = vat.where(vat.ge(0) & vat.le(1), default_vat_rate)
            vat = vat.fillna(default_vat_rate)
        else:
            vat = pd.Series(
                default_vat_rate,
                index=result.index,
                dtype="float64",
            )

        result["vat_rate_used"] = vat

        # Gross amount = Net amount * (1 + VAT rate)
        result["net_spend"] = amount / (1 + vat)

    result["amount_parse_failed"] = amount.isna()

    return result


def calculate_spend_total(
    df: pd.DataFrame,
    spend_column: str = "net_spend",
    include_column: str | None = None,
) -> float:
    """Calculate total spend, optionally considering included records only."""

    if spend_column not in df.columns:
        raise KeyError(f"Spend column not found: {spend_column}")

    data = df.copy()

    if include_column and include_column in data.columns:
        data = data.loc[data[include_column].eq(1)]

    return float(
        pd.to_numeric(data[spend_column], errors="coerce")
        .fillna(0)
        .sum()
    )
