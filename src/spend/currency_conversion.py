
"""Convert spend using monthly FX rates."""

from __future__ import annotations

import pandas as pd


def convert_to_reporting_currency(
    df: pd.DataFrame,
    amount_column: str,
    currency_column: str,
    reporting_currency: str,
    date_column: str | None = None,
    rate_column: str | None = None,
    fx_rates: pd.DataFrame | None = None,
    fx_currency_column: str = "currency",
    fx_month_column: str = "month",
    fx_rate_column: str = "units_per_usd",
) -> pd.DataFrame:
    """
    Convert amounts into the reporting currency.

    Supported FX table convention:
        units_per_usd = local currency units per 1 USD.

    For USD reporting:
        USD amount = local amount / units_per_usd.

    If source and reporting currencies are the same, no conversion
    is needed.

    Missing rates are flagged rather than guessed.
    """
    required = [amount_column, currency_column]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(f"Required columns not found: {missing}")

    result = df.copy()

    reporting_currency = reporting_currency.upper().strip()

    amount = pd.to_numeric(result[amount_column], errors="coerce")
    currency = (
        result[currency_column]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    result["source_currency"] = currency
    result["reporting_currency"] = reporting_currency

    result["exchange_rate_used"] = pd.NA
    result["converted_spend"] = pd.NA
    result["fx_missing"] = False

    # No conversion required when currencies already match.
    same_currency = currency.eq(reporting_currency)

    result.loc[same_currency, "exchange_rate_used"] = 1.0
    result.loc[same_currency, "converted_spend"] = amount[same_currency]

    needs_conversion = ~same_currency

    if not needs_conversion.any():
        result["fx_missing"] = result["converted_spend"].isna()
        return result

    if reporting_currency != "USD":
        raise NotImplementedError(
            "The supplied FX convention supports conversion to USD. "
            "For another reporting currency, provide cross-rate logic."
        )

    if fx_rates is None:
        result.loc[needs_conversion, "fx_missing"] = True
        return result

    fx_required = [
        fx_currency_column,
        fx_month_column,
        fx_rate_column,
    ]

    missing_fx_columns = [
        c for c in fx_required if c not in fx_rates.columns
    ]

    if missing_fx_columns:
        raise KeyError(
            f"FX columns not found: {missing_fx_columns}"
        )

    fx = fx_rates[fx_required].copy()

    fx[fx_currency_column] = (
        fx[fx_currency_column].astype(str).str.upper().str.strip()
    )

    fx[fx_month_column] = (
        fx[fx_month_column].astype(str).str[:7]
    )

    fx[fx_rate_column] = pd.to_numeric(
        fx[fx_rate_column],
        errors="coerce",
    )

    # Only valid positive rates are usable.
    fx = fx.loc[fx[fx_rate_column].gt(0)]

    # Avoid silently choosing between conflicting rates.
    conflicts = (
        fx.groupby([fx_currency_column, fx_month_column])[fx_rate_column]
        .nunique()
    )
    if conflicts.gt(1).any():
        raise ValueError(
            "FX table contains conflicting rates for the same "
            "currency and month."
        )

    fx = fx.drop_duplicates(
        subset=[fx_currency_column, fx_month_column]
    )

    if date_column and date_column in result.columns:
        dates = pd.to_datetime(
            result[date_column],
            errors="coerce",
            dayfirst=True,
        )
        months = dates.dt.strftime("%Y-%m")
    else:
        months = pd.Series("", index=result.index)

    # Build monthly FX lookup.
    fx_lookup = {
        (row[fx_currency_column], row[fx_month_column]):
        float(row[fx_rate_column])
        for _, row in fx.iterrows()
    }

    for index in result.index[needs_conversion]:
        key = (currency.loc[index], months.loc[index])
        rate = fx_lookup.get(key)

        if rate is None or pd.isna(amount.loc[index]):
            result.at[index, "fx_missing"] = True
            continue

        result.at[index, "exchange_rate_used"] = rate
        result.at[index, "converted_spend"] = amount.loc[index] / rate

    result["fx_missing"] = result["converted_spend"].isna()

    return result
