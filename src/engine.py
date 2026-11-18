
"""
Company-specific processing engines for the Cantill Spend Intelligence task.

Company A:
    ERP goods receipts + invoice-tool exports, reporting in ZAR.

Company B:
    Multi-currency invoice lines + FX rates + seed labels, reporting in USD.

Company C:
    Generic blind-test invoice processing using YAML column mappings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .classifier import TaxonomyClassifier
from .config import load_config
from .utils import (
    OUTPUT_COLUMNS,
    first_non_empty,
    norm_id,
    norm_text,
    parse_amount,
    parse_dates,
    prepare_output,
    require_columns,
    stable_key,
)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

NON_PROCUREMENT_TERMS = [
    "intercompany",
    "inter company",
    "royalty",
    "dividend",
    "income tax",
    "vat payable",
    "withholding tax",
    "corporate tax",
    "revenue",
    "customer rebate",
    "partner billing",
    "interest expense",
    "loan repayment",
    "financing cost",
]


def _read_csv(path: str | Path, name: str) -> pd.DataFrame:
    """Read a CSV file and provide a useful error message."""
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"{name} file not found: {file_path}"
        )

    try:
        return pd.read_csv(file_path, low_memory=False)
    except Exception as exc:
        raise ValueError(
            f"Could not read {name} CSV: {file_path}. {exc}"
        ) from exc


def _require(df: pd.DataFrame, columns: list[str], source: str) -> None:
    """Validate required columns."""
    require_columns(df, columns, source)


def _optional_series(
    df: pd.DataFrame,
    column: str | None,
    default: str = "",
) -> pd.Series:
    """Return a source column or a default-filled Series."""
    if column and column in df.columns:
        return df[column]

    return pd.Series(default, index=df.index, dtype="object")


def _non_procurement_mask(series: pd.Series) -> pd.Series:
    """Identify descriptions that indicate non-procurement spend."""
    normalized = series.fillna("").astype(str).map(norm_text)

    pattern = "|".join(
        term.replace(" ", r"\s+")
        for term in NON_PROCUREMENT_TERMS
    )

    return normalized.str.contains(pattern, regex=True, na=False)


def _taxonomy_classifier(path: str) -> TaxonomyClassifier:
    """Load taxonomy and create the classifier."""
    taxonomy = _read_csv(path, "Taxonomy").fillna("")

    if taxonomy.empty:
        raise ValueError(f"Taxonomy file is empty: {path}")

    category_columns = [
        column
        for column in taxonomy.columns
        if column.lower() in {
            "category_l1",
            "category_l2",
            "category_l3",
            "category_l4",
        }
    ]

    if not category_columns:
        raise ValueError(
            "Taxonomy must contain category_l1, category_l2, "
            "category_l3, or category_l4 columns."
        )

    return TaxonomyClassifier(taxonomy)


def _path_columns(paths: list[list[str]]) -> dict[str, list[str]]:
    """Convert predicted taxonomy paths into L1-L4 output columns."""
    result = {}

    for level in range(4):
        result[f"l{level + 1}"] = [
            path[level] if len(path) > level else ""
            for path in paths
        ]

    return result


def _make_output(
    record_ids,
    include,
    reasons,
    supplier_keys,
    item_keys,
    spend,
    paths,
    confidence,
) -> pd.DataFrame:
    """Build the standard assignment output."""
    output = pd.DataFrame({
        "record_id": record_ids,
        "include": include,
        "exclusion_reason": reasons,
        "supplier_key": supplier_keys,
        "item_key": item_keys,
        "spend": spend,
        **_path_columns(paths),
        "confidence": confidence,
    })

    return finalize(output)


def _build_fx_map(fx: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Build a currency/month to units-per-USD lookup."""
    _require(
        fx,
        ["currency", "month", "units_per_usd"],
        "FX rates",
    )

    rates = {}

    for row in fx.itertuples(index=False):
        currency = str(row.currency).strip().upper()
        month = str(row.month).strip()[:7]

        try:
            rate = float(row.units_per_usd)
        except (TypeError, ValueError):
            continue

        if currency and month and np.isfinite(rate) and rate > 0:
            rates[(currency, month)] = rate

    return rates


def _convert_to_usd(
    amounts: pd.Series,
    currencies: pd.Series,
    row_rates: pd.Series,
    dates: pd.Series,
    fx_map: dict[tuple[str, str], float] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Convert amounts into USD.

    FX convention:
        local currency amount / units of local currency per USD = USD amount.

    USD rows always use a rate of 1.
    """
    amounts = pd.to_numeric(amounts, errors="coerce").fillna(0.0)

    currencies = (
        currencies.fillna("USD")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    rates = pd.to_numeric(row_rates, errors="coerce").copy()

    rates.loc[currencies.eq("USD")] = 1.0

    months = parse_dates(dates).dt.strftime("%Y-%m")

    if fx_map:
        for index in amounts.index:
            currency = currencies.loc[index]

            if currency == "USD":
                continue

            current_rate = rates.loc[index]

            if pd.isna(current_rate) or current_rate <= 0:
                rates.loc[index] = fx_map.get(
                    (currency, months.loc[index]),
                    np.nan,
                )

    rates = rates.where(rates.gt(0), np.nan)

    converted = amounts.div(rates)
    converted = converted.replace([np.inf, -np.inf], np.nan)

    return converted.fillna(0.0), rates


def _supplier_key(identifier, name="") -> str:
    """Prefer supplier identifier; fall back to supplier name."""
    value = first_non_empty(identifier, name)
    return stable_key("SUP", value)


def _item_key(identifier, description="") -> str:
    """Prefer item identifier; fall back to item description."""
    value = first_non_empty(identifier, description)
    return stable_key("ITEM", value)


# ---------------------------------------------------------------------
# Company A
# ---------------------------------------------------------------------

def run_a(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Process Company A.

    ERP amounts are treated as net ZAR.
    Invoice-tool gross amounts are converted to net amounts using VAT.
    """
    inputs = cfg["inputs"]

    erp = _read_csv(inputs["erp"], "Company A ERP")
    tool = _read_csv(inputs["tool"], "Company A invoice tool")
    labels = _read_csv(inputs["labels"], "Company A labels")
    classifier = _taxonomy_classifier(inputs["taxonomy"])

    _require(
        erp,
        [
            "line_id",
            "value_zar",
            "material_short_text",
            "service_short_text",
            "vendor_id",
            "supplier_name",
            "material_number",
            "po_number",
        ],
        "Company A ERP",
    )

    _require(
        tool,
        [
            "tool_record_id",
            "invoice_number",
            "supplier_name_as_captured",
            "po_reference",
            "line_text",
            "gross_amount",
            "vat_rate",
            "status",
        ],
        "Company A invoice tool",
    )

    _require(
        labels,
        ["record_id"],
        "Company A labels",
    )

    # ERP records.
    e = erp.copy()

    e["record_id"] = "ERP:" + e["line_id"].astype(str)
    e["spend"] = parse_amount(e["value_zar"])

    e["description"] = (
        e["material_short_text"].fillna("").astype(str)
        + " "
        + e["service_short_text"].fillna("").astype(str)
    ).str.strip()

    e["supplier_key"] = [
        _supplier_key(vendor, name)
        for vendor, name in zip(
            e["vendor_id"],
            e["supplier_name"],
        )
    ]

    e["item_key"] = [
        _item_key(material, description)
        for material, description in zip(
            e["material_number"],
            e["material_short_text"],
        )
    ]

    e["include"] = 1
    e["reason"] = ""

    non_proc = _non_procurement_mask(e["description"])

    e.loc[non_proc, "include"] = 0
    e.loc[non_proc, "reason"] = "NON_PROCUREMENT"

    erp_paths, erp_confidence = classifier.predict(
        e["description"].fillna("").astype(str).tolist()
    )

    # Invoice-tool records.
    t = tool.copy()

    t["record_id"] = "TOOL:" + t["tool_record_id"].astype(str)

    gross = parse_amount(t["gross_amount"])

    vat = pd.to_numeric(
        t["vat_rate"].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )

    vat = vat.where(vat <= 1, vat / 100)
    vat = vat.fillna(0.15).clip(lower=0)

    t["spend"] = gross / (1 + vat)

    t["description"] = (
        t["line_text"].fillna("").astype(str)
    )

    status = (
        t["status"].fillna("").astype(str).str.strip().str.lower()
    )

    t["include"] = status.isin(["approved", "paid"]).astype(int)
    t["reason"] = np.where(
        t["include"].eq(1),
        "",
        "NOT_APPROVED",
    )

    t.loc[status.eq("cancelled"), "reason"] = "CANCELLED"

    t["supplier_key"] = [
        _supplier_key("", name)
        for name in t["supplier_name_as_captured"]
    ]

    t["item_key"] = [
        _item_key("", description)
        for description in t["description"]
    ]

    # Match supplier names to ERP supplier names exactly after normalization.
    supplier_lookup = {}

    for name, key in zip(e["supplier_name"], e["supplier_key"]):
        normalized = norm_text(name)

        if normalized:
            supplier_lookup.setdefault(normalized, key)

    for index, name in t["supplier_name_as_captured"].items():
        normalized = norm_text(name)

        if normalized in supplier_lookup:
            t.at[index, "supplier_key"] = supplier_lookup[normalized]

    # Match invoice-tool records to ERP records using PO and amount.
    e_po = e["po_number"].map(norm_id)
    t_po = t["po_reference"].map(norm_id)

    erp_groups = {
        po: list(group.index)
        for po, group in e[e_po.ne("")].groupby(e_po[e_po.ne("")])
    }

    used_erp_indices = set()

    for tool_index in t.index[t["include"].eq(1)]:
        po = t_po.loc[tool_index]

        if not po:
            continue

        candidates = [
            index
            for index in erp_groups.get(po, [])
            if index not in used_erp_indices
            and int(e.at[index, "include"]) == 1
        ]

        if not candidates:
            continue

        differences = (
            e.loc[candidates, "spend"] - t.at[tool_index, "spend"]
        ).abs()

        best_index = differences.idxmin()
        erp_amount = abs(float(e.at[best_index, "spend"]))

        denominator = max(erp_amount, 1.0)
        relative_difference = (
            float(differences.loc[best_index]) / denominator
        )

        if relative_difference <= 0.03:
            t.at[tool_index, "include"] = 0
            t.at[tool_index, "reason"] = "DUPLICATE"
            used_erp_indices.add(best_index)

    # Remove repeated invoice-tool exports.
    invoice_norm = t["invoice_number"].map(norm_id)

    duplicate_key = (
        invoice_norm
        + "|"
        + t["supplier_key"].astype(str)
        + "|"
        + t["spend"].round(2).astype(str)
    )

    has_invoice = invoice_norm.ne("")

    duplicate_mask = (
        has_invoice
        & duplicate_key.duplicated(keep="first")
        & t["include"].eq(1)
    )

    t.loc[duplicate_mask, "include"] = 0
    t.loc[duplicate_mask, "reason"] = "DUPLICATE"

    tool_paths, tool_confidence = classifier.predict(
        t["description"].fillna("").astype(str).tolist()
    )

    erp_output = _make_output(
        record_ids=e["record_id"],
        include=e["include"],
        reasons=e["reason"],
        supplier_keys=e["supplier_key"],
        item_keys=e["item_key"],
        spend=np.where(e["include"].eq(1), e["spend"], 0),
        paths=erp_paths,
        confidence=erp_confidence,
    )

    tool_output = _make_output(
        record_ids=t["record_id"],
        include=t["include"],
        reasons=t["reason"],
        supplier_keys=t["supplier_key"],
        item_keys=t["item_key"],
        spend=np.where(t["include"].eq(1), t["spend"], 0),
        paths=tool_paths,
        confidence=tool_confidence,
    )

    return finalize(
        pd.concat([erp_output, tool_output], ignore_index=True)
    )


# ---------------------------------------------------------------------
# Company B
# ---------------------------------------------------------------------

def run_b(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Process Company B.

    Converts invoice amounts into USD, applies seed labels,
    detects duplicates, and classifies spend using the taxonomy.
    """
    inputs = cfg["inputs"]

    invoices = _read_csv(inputs["invoices"], "Company B invoices")
    fx = _read_csv(inputs["fx"], "Company B FX rates")
    labels = _read_csv(inputs["labels"], "Company B seed labels")
    classifier = _taxonomy_classifier(inputs["taxonomy"])

    _require(
        invoices,
        [
            "line_id",
            "line_amount",
            "currency",
            "currency_rate",
            "invoice_date",
            "posting_date",
            "line_description",
            "vendor_id",
            "material_number",
            "gl_account_description",
            "gl_account",
            "document_type",
            "source_system",
            "company_code",
            "invoice_number",
            "invoice_line",
        ],
        "Company B invoices",
    )

    _require(
        labels,
        [
            "line_id",
            "include",
            "exclusion_reason",
            "category_l1",
            "category_l2",
            "category_l3",
        ],
        "Company B seed labels",
    )

    d = invoices.copy()

    d["record_id"] = d["line_id"].astype(str)

    amounts = parse_amount(d["line_amount"])

    currencies = (
        d["currency"].fillna("USD").astype(str).str.upper()
    )

    dates = parse_dates(d["invoice_date"]).fillna(
        parse_dates(d["posting_date"])
    )

    fx_map = _build_fx_map(fx)

    converted, resolved_rates = _convert_to_usd(
        amounts=amounts,
        currencies=currencies,
        row_rates=d["currency_rate"],
        dates=dates,
        fx_map=fx_map,
    )

    d["spend"] = converted
    d["description"] = (
        d["line_description"].fillna("").astype(str)
    )

    d["supplier_key"] = [
        _supplier_key(vendor)
        for vendor in d["vendor_id"]
    ]

    d["item_key"] = [
        _item_key(material, description)
        for material, description in zip(
            d["material_number"],
            d["description"],
        )
    ]

    # Train the classifier using available seed labels.
    line_text = d.set_index(
        d["line_id"].astype(str)
    )["description"]

    label_ids = labels["line_id"].astype(str)

    training_text = line_text.reindex(label_ids).fillna("").tolist()

    label_categories = labels[
        ["category_l3", "category_l2", "category_l1"]
    ].fillna("").astype(str)

    training_labels = []

    for row in label_categories.itertuples(index=False, name=None):
        category = first_non_empty(*row)
        training_labels.append(category)

    classifier.fit(training_text, training_labels)

    paths, confidence = classifier.predict(
        d["description"].tolist()
    )

    # Default inclusion decision.
    d["include"] = 1
    d["reason"] = ""

    non_proc = _non_procurement_mask(d["description"])

    d.loc[non_proc, "include"] = 0
    d.loc[non_proc, "reason"] = "NON_PROCUREMENT"

    gl_text = (
        d["gl_account_description"].fillna("").astype(str)
        + " "
        + d["gl_account"].fillna("").astype(str)
    )

    gl_normalized = gl_text.map(norm_text)

    gl_non_proc = gl_normalized.str.contains(
        r"intercompany|ic payable|income tax|vat payable|"
        r"royalt|revenue|financing",
        regex=True,
        na=False,
    )

    d.loc[gl_non_proc, "include"] = 0
    d.loc[gl_non_proc, "reason"] = "NON_PROCUREMENT"

    document_type = (
        d["document_type"].fillna("").astype(str).str.upper()
    )

    cancelled = document_type.eq("CA")

    d.loc[cancelled, "include"] = 0
    d.loc[cancelled, "reason"] = "CANCELLED"

    # Seed labels are authoritative for their exact line IDs.
    seed_map = {}

    for row in labels.itertuples(index=False):
        line_id = str(row.line_id)

        try:
            include_value = int(row.include)
        except (TypeError, ValueError):
            continue

        if include_value not in (0, 1):
            continue

        reason = (
            ""
            if pd.isna(row.exclusion_reason)
            else str(row.exclusion_reason).strip()
        )

        seed_map[line_id] = (include_value, reason)

    for index, line_id in d["line_id"].astype(str).items():
        if line_id in seed_map:
            include_value, reason = seed_map[line_id]
            d.at[index, "include"] = include_value
            d.at[index, "reason"] = reason

    # Exact duplicate invoice lines only.
    duplicate_columns = [
        "source_system",
        "company_code",
        "invoice_number",
        "invoice_line",
        "vendor_id",
        "line_amount",
        "currency",
    ]

    duplicate_mask = (
        d.duplicated(duplicate_columns, keep="first")
        & d["include"].eq(1)
    )

    d.loc[duplicate_mask, "include"] = 0
    d.loc[duplicate_mask, "reason"] = "DUPLICATE"

    # Do not invent spend if an FX rate cannot be resolved.
    unresolved_fx = (
        currencies.ne("USD")
        & resolved_rates.isna()
        & amounts.ne(0)
    )

    d.loc[unresolved_fx, "include"] = 0
    d.loc[unresolved_fx, "reason"] = "NON_PROCUREMENT"

    return _make_output(
        record_ids=d["record_id"],
        include=d["include"],
        reasons=d["reason"],
        supplier_keys=d["supplier_key"],
        item_keys=d["item_key"],
        spend=np.where(d["include"].eq(1), d["spend"], 0),
        paths=paths,
        confidence=confidence,
    )


# ---------------------------------------------------------------------
# Company C
# ---------------------------------------------------------------------

def run_c(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Process Company C blind-test records.

    The source column names are supplied through the YAML 'columns' mapping.
    No labelled training data is assumed.
    """
    inputs = cfg["inputs"]

    d = _read_csv(inputs["invoices"], "Company C invoices")
    classifier = _taxonomy_classifier(inputs["taxonomy"])

    mapping = cfg.get("columns", {})

    def get_column(name: str, default=None) -> pd.Series:
        source_column = mapping.get(name, default)

        if source_column and source_column in d.columns:
            return d[source_column]

        return pd.Series("", index=d.index, dtype="object")

    # Record identifier is required.
    id_column = mapping.get("record_id", "line_id")

    if id_column not in d.columns:
        raise ValueError(
            f"Company C record ID column '{id_column}' was not found. "
            f"Available columns: {list(d.columns)}"
        )

    record_ids = d[id_column].astype(str)

    if record_ids.str.strip().eq("").any():
        raise ValueError("Company C contains blank record IDs.")

    raw_amount = parse_amount(
        get_column("amount")
    )

    default_currency = cfg.get("reporting_currency", "USD")

    currency_series = get_column("currency")

    currency_series = currency_series.where(
        currency_series.astype(str).str.strip().ne(""),
        default_currency,
    )

    currencies = (
        currency_series.fillna(default_currency)
        .astype(str)
        .str.upper()
    )

    row_rates = pd.to_numeric(
        get_column("currency_rate"),
        errors="coerce",
    )

    dates = parse_dates(get_column("date"))

    fx_map = None
    fx_path = inputs.get("fx")

    if fx_path:
        fx_file = Path(fx_path)

        if fx_file.is_file():
            fx = _read_csv(fx_file, "Company C FX rates")
            fx_map = _build_fx_map(fx)

    spend, resolved_rates = _convert_to_usd(
        amounts=raw_amount,
        currencies=currencies,
        row_rates=row_rates,
        dates=dates,
        fx_map=fx_map,
    )

    descriptions = (
        get_column("description")
        .fillna("")
        .astype(str)
    )

    supplier_ids = (
        get_column("supplier_id")
        .fillna("")
        .astype(str)
    )

    supplier_names = (
        get_column("supplier_name")
        .fillna("")
        .astype(str)
    )

    if supplier_ids.str.strip().eq("").all():
        supplier_values = supplier_names
    else:
        supplier_values = supplier_ids.where(
            supplier_ids.str.strip().ne(""),
            supplier_names,
        )

    supplier_keys = supplier_values.map(
        lambda value: stable_key("SUP", value)
    )

    item_ids = (
        get_column("item_id")
        .fillna("")
        .astype(str)
    )

    item_values = item_ids.where(
        item_ids.str.strip().ne(""),
        descriptions,
    )

    item_keys = item_values.map(
        lambda value: stable_key("ITEM", value)
    )

    paths, confidence = classifier.predict(
        descriptions.tolist()
    )

    include = pd.Series(1, index=d.index, dtype=int)
    reasons = pd.Series("", index=d.index, dtype="object")

    # Apply status rules only if a status column is provided.
    status = (
        get_column("status")
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if status.str.strip().ne("").any():
        allowed = status.isin(
            ["approved", "paid", "posted", "completed"]
        )

        cancelled = status.str.contains(
            "cancel", na=False
        )

        include.loc[~allowed] = 0
        reasons.loc[~allowed] = "NOT_APPROVED"

        reasons.loc[cancelled] = "CANCELLED"

    # Exclude non-procurement descriptions.
    non_proc = _non_procurement_mask(descriptions)

    include.loc[non_proc] = 0
    reasons.loc[non_proc] = "NON_PROCUREMENT"

    gl_description = (
        get_column("gl_account_description")
        .fillna("")
        .astype(str)
        .map(norm_text)
    )

    gl_non_proc = gl_description.str.contains(
        r"intercompany|ic payable|income tax|vat payable|"
        r"royalt|revenue|financing",
        regex=True,
        na=False,
    )

    include.loc[gl_non_proc] = 0
    reasons.loc[gl_non_proc] = "NON_PROCUREMENT"

    # Exact duplicate invoice lines, where sufficient identifiers exist.
    duplicate_columns = []

    for mapping_name in [
        "invoice_number",
        "invoice_line",
        "supplier_id",
        "amount",
        "currency",
    ]:
        source_column = mapping.get(mapping_name)

        if source_column and source_column in d.columns:
            duplicate_columns.append(source_column)

    if len(duplicate_columns) >= 3:
        duplicate_mask = (
            d.duplicated(duplicate_columns, keep="first")
            & include.eq(1)
        )

        include.loc[duplicate_mask] = 0
        reasons.loc[duplicate_mask] = "DUPLICATE"

    # Exclude non-USD records when no valid conversion rate is available.
    unresolved_fx = (
        currencies.ne("USD")
        & resolved_rates.isna()
        & raw_amount.ne(0)
    )

    include.loc[unresolved_fx] = 0
    reasons.loc[unresolved_fx] = "NON_PROCUREMENT"

    return _make_output(
        record_ids=record_ids,
        include=include,
        reasons=reasons,
        supplier_keys=supplier_keys,
        item_keys=item_keys,
        spend=np.where(include.eq(1), spend, 0),
        paths=paths,
        confidence=confidence,
    )


# ---------------------------------------------------------------------
# Shared output finalization
# ---------------------------------------------------------------------

def finalize(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Enforce the exact output template and business rules."""
    output = dataframe.copy()

    for column in OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = ""

    output = output[OUTPUT_COLUMNS]

    output["include"] = pd.to_numeric(
        output["include"], errors="coerce"
    ).fillna(0).astype(int)

    output["spend"] = pd.to_numeric(
        output["spend"], errors="coerce"
    ).fillna(0).round(2)

    output["confidence"] = pd.to_numeric(
        output["confidence"], errors="coerce"
    ).fillna(0).clip(0, 1).round(4)

    for column in [
        "record_id",
        "exclusion_reason",
        "supplier_key",
        "item_key",
        "l1",
        "l2",
        "l3",
        "l4",
    ]:
        output[column] = (
            output[column]
            .fillna("")
            .astype(str)
        )

    # Excluded records must have zero spend.
    output.loc[
        output["include"].eq(0), "spend"
    ] = 0.0

    # Included records must not carry an exclusion reason.
    output.loc[
        output["include"].eq(1), "exclusion_reason"
    ] = ""

    return prepare_output(output)


# ---------------------------------------------------------------------
# Backward-compatible execution helper
# ---------------------------------------------------------------------

def execute(config_path: str | Path):
    """Load a company configuration, execute its engine, and save output."""
    config = load_config(config_path, check_input_files=True)

    company = config["company"]

    runners = {
        "A": run_a,
        "B": run_b,
        "C": run_c,
    }

    result = runners[company](config)

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)

    return output_path, result
