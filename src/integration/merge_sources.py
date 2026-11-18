
"""
Module: merge_sources.py

Purpose:
    Integrate purchase records from multiple company systems.

Main use:
    Company A:
        1. ERP goods receipts.
        2. Invoice tool exports.

Responsibilities:
    - Standardize purchase order references.
    - Match ERP and invoice-tool records.
    - Compare net purchase amounts.
    - Identify cross-system duplicates.
    - Detect repeated invoice exports.
    - Preserve all original records.
    - Generate integration summary.

Important:
    ERP records are retained as the primary records.
    Matched invoice-tool records are marked DUPLICATE.
    No original records are deleted.
"""

import re
import numpy as np
import pandas as pd

from difflib import SequenceMatcher


# --------------------------------------------------
# 1. Normalize Purchase Order References
# --------------------------------------------------

def normalize_po_reference(value):
    """
    Extract and normalize a purchase order number.

    Examples:
        'PO 4510000018' -> '4510000018'
        '4510000018/10.0' -> '4510000018'
        '4510000018' -> '4510000018'

    Returns:
        Normalized PO number.
    """

    if pd.isna(value):
        return ""

    text = str(value).upper().strip()

    # Extract numeric PO identifier.
    match = re.search(r"\d{7,}", text)

    if not match:
        return ""

    return match.group(0)


# --------------------------------------------------
# 2. Extract PO Item Number
# --------------------------------------------------

def extract_po_item(value):
    """
    Extract an optional PO item number.

    Example:
        '4510000018/10.0' -> '10'
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    match = re.search(
        r"/\s*(\d+(?:\.0+)?)\s*$",
        text
    )

    if not match:
        return ""

    return str(int(float(match.group(1))))


def normalize_po_item(value):
    """Normalize ERP PO item numbers."""

    if pd.isna(value):
        return ""

    try:
        return str(int(float(value)))
    except (ValueError, TypeError, OverflowError):
        return str(value).strip()


# --------------------------------------------------
# 3. Normalize Supplier Names
# --------------------------------------------------

def normalize_supplier_name(value):
    """
    Normalize supplier names for matching.
    """

    if pd.isna(value):
        return ""

    text = str(value).lower().strip()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def supplier_similarity(name_a, name_b):
    """
    Calculate supplier-name similarity.

    Returns:
        Similarity score between 0 and 1.
    """

    a = normalize_supplier_name(name_a)
    b = normalize_supplier_name(name_b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------
# 4. Calculate Net Amount
# --------------------------------------------------

def get_net_amount(
    df,
    source,
    amount_column=None,
    vat_column="vat_rate",
    default_vat=0.15
):
    """
    Get net purchase amount.

    ERP:
        Values are already net of tax.

    Invoice tool:
        Uses net_spend if previously calculated.
        Otherwise calculates net from gross amount.
    """

    if source == "ERP":

        column = amount_column or "value_zar"

        if column not in df.columns:
            raise ValueError(
                f"ERP amount column not found: {column}"
            )

        return pd.to_numeric(
            df[column],
            errors="coerce"
        )

    if source == "TOOL":

        if "net_spend" in df.columns:
            return pd.to_numeric(
                df["net_spend"],
                errors="coerce"
            )

        column = amount_column or "gross_amount"

        if column not in df.columns:
            raise ValueError(
                f"Tool amount column not found: {column}"
            )

        gross = pd.to_numeric(
            df[column].astype(str).str.replace(
                ",", "", regex=False
            ),
            errors="coerce"
        )

        if vat_column not in df.columns:
            vat = pd.Series(
                default_vat,
                index=df.index
            )
        else:
            raw_vat = df[vat_column].astype("string").str.strip()

            percentage = raw_vat.str.endswith("%", na=False)

            vat = pd.to_numeric(
                raw_vat.str.replace("%", "", regex=False),
                errors="coerce"
            )

            vat = vat.where(
                ~percentage,
                vat / 100
            )

            vat = vat.where(
                vat <= 1,
                vat / 100
            )

            vat = vat.fillna(default_vat)

        return gross / (1 + vat)

    raise ValueError(
        "Source must be ERP or TOOL."
    )


# --------------------------------------------------
# 5. Prepare ERP Records
# --------------------------------------------------

def prepare_erp_data(erp_df):
    """
    Prepare ERP records for integration.
    """

    erp = erp_df.copy().reset_index(drop=True)

    required_columns = [
        "line_id",
        "po_number",
        "value_zar"
    ]

    missing = set(required_columns) - set(erp.columns)

    if missing:
        raise ValueError(
            f"ERP missing required columns: {sorted(missing)}"
        )

    erp["record_id"] = (
        "ERP:" + erp["line_id"].astype(str)
    )

    erp["po_normalized"] = (
        erp["po_number"].apply(normalize_po_reference)
    )

    erp["po_item_normalized"] = (
        erp["po_item"].apply(normalize_po_item)
        if "po_item" in erp.columns
        else ""
    )

    erp["net_amount"] = get_net_amount(
        erp,
        source="ERP"
    )

    erp["source_type"] = "ERP"

    erp["integration_status"] = "ERP_PRIMARY"

    erp["matched_record_id"] = ""

    erp["match_method"] = ""

    erp["match_score"] = 0.0

    return erp


# --------------------------------------------------
# 6. Prepare Invoice Tool Records
# --------------------------------------------------

def prepare_tool_data(tool_df):
    """
    Prepare invoice-tool records for integration.
    """

    tool = tool_df.copy().reset_index(drop=True)

    required_columns = [
        "tool_record_id",
        "po_reference",
        "gross_amount"
    ]

    missing = set(required_columns) - set(tool.columns)

    if missing:
        raise ValueError(
            f"Tool data missing required columns: {sorted(missing)}"
        )

    tool["record_id"] = (
        "TOOL:" + tool["tool_record_id"].astype(str)
    )

    tool["po_normalized"] = (
        tool["po_reference"].apply(
            normalize_po_reference
        )
    )

    tool["po_item_normalized"] = (
        tool["po_reference"].apply(
            extract_po_item
        )
    )

    tool["net_amount"] = get_net_amount(
        tool,
        source="TOOL"
    )

    tool["source_type"] = "TOOL"

    tool["integration_status"] = "TOOL_ONLY"

    tool["matched_record_id"] = ""

    tool["match_method"] = ""

    tool["match_score"] = 0.0

    return tool


# --------------------------------------------------
# 7. Detect Repeated Invoice Exports
# --------------------------------------------------

def detect_tool_duplicates(tool_df):
    """
    Identify repeated invoice-tool exports.

    Matching fields:
        - Invoice number
        - Supplier name
        - Net amount
        - Line description

    The first eligible record is retained.

    Other records are marked as duplicate.
    """

    tool = tool_df.copy()

    required_columns = [
        "invoice_number",
        "supplier_name_as_captured",
        "line_text"
    ]

    missing = [
        c for c in required_columns
        if c not in tool.columns
    ]

    if missing:
        raise ValueError(
            f"Missing duplicate detection columns: {missing}"
        )

    tool["_invoice_normalized"] = (
        tool["invoice_number"].apply(
            normalize_po_reference
        )
    )

    # Invoice numbers may be alphanumeric.
    tool["_invoice_normalized"] = (
        tool["invoice_number"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

    tool["_supplier_normalized"] = (
        tool["supplier_name_as_captured"].apply(
            normalize_supplier_name
        )
    )

    tool["_description_normalized"] = (
        tool["line_text"].apply(
            normalize_supplier_name
        )
    )

    duplicate_columns = [
        "_invoice_normalized",
        "_supplier_normalized",
        "net_amount",
        "_description_normalized"
    ]

    # Missing invoice or supplier identifiers should not
    # cause unrelated records to be grouped together.
    valid_key = (
        tool["_invoice_normalized"].ne("")
        & tool["_supplier_normalized"].ne("")
    )

    duplicate_mask = (
        tool.loc[valid_key]
        .duplicated(
            subset=duplicate_columns,
            keep="first"
        )
    )

    tool["_tool_duplicate"] = False

    tool.loc[valid_key, "_tool_duplicate"] = (
        duplicate_mask
    )

    tool.loc[
        tool["_tool_duplicate"],
        "integration_status"
    ] = "TOOL_EXPORT_DUPLICATE"

    tool.loc[
        tool["_tool_duplicate"],
        "match_method"
    ] = "INVOICE_EXPORT_DUPLICATE"

    tool.drop(
        columns=[
            "_invoice_normalized",
            "_supplier_normalized",
            "_description_normalized"
        ],
        inplace=True
    )

    return tool


# --------------------------------------------------
# 8. Match ERP and Invoice Tool Records
# --------------------------------------------------

def match_erp_tool_records(
    erp_df,
    tool_df,
    amount_tolerance=0.03,
    absolute_tolerance=1.0,
    supplier_threshold=0.90
):
    """
    Match purchase records across ERP and invoice tool.

    Matching logic:

        1. Normalize PO numbers.
        2. Find ERP candidates with the same PO.
        3. Compare PO item when available.
        4. Compare net amounts.
        5. Compare supplier names when available.
        6. Select the best unused ERP candidate.

    A successful match marks the tool record as
    TOOL_MATCHED_DUPLICATE.

    All records are preserved.
    """

    erp = erp_df.copy()

    tool = tool_df.copy()

    erp["match_used"] = False

    # Index ERP records by normalized PO.
    po_groups = {
        po: group.index.tolist()
        for po, group in erp[
            erp["po_normalized"].ne("")
        ].groupby("po_normalized")
    }

    for tool_index, tool_row in tool.iterrows():

        # Skip repeated tool exports.
        if (
            tool_row["integration_status"]
            == "TOOL_EXPORT_DUPLICATE"
        ):
            continue

        tool_po = tool_row["po_normalized"]

        if not tool_po:
            continue

        candidates = po_groups.get(tool_po, [])

        candidates = [
            idx for idx in candidates
            if not erp.at[idx, "match_used"]
        ]

        if not candidates:
            continue

        valid_candidates = []

        for erp_index in candidates:

            erp_row = erp.loc[erp_index]

            erp_amount = erp_row["net_amount"]

            tool_amount = tool_row["net_amount"]

            if pd.isna(erp_amount) or pd.isna(tool_amount):
                continue

            amount_difference = abs(
                float(erp_amount) - float(tool_amount)
            )

            amount_limit = max(
                absolute_tolerance,
                abs(float(erp_amount)) * amount_tolerance
            )

            if amount_difference > amount_limit:
                continue

            # PO item comparison.
            item_score = 1.0

            tool_item = tool_row["po_item_normalized"]

            erp_item = erp_row["po_item_normalized"]

            if tool_item and erp_item:
                item_score = float(tool_item == erp_item)

            # Supplier comparison, when both names exist.
            supplier_score = 0.0

            if (
                "supplier_name" in erp.columns
                and "supplier_name_as_captured" in tool.columns
            ):

                supplier_score = supplier_similarity(
                    erp_row["supplier_name"],
                    tool_row["supplier_name_as_captured"]
                )

                if supplier_score < supplier_threshold:
                    continue

            # Lower amount difference is better.
            amount_score = 1 - min(
                amount_difference / max(amount_limit, 1.0),
                1.0
            )

            score = (
                0.50 * amount_score
                + 0.25 * item_score
                + 0.25 * supplier_score
            )

            valid_candidates.append(
                (
                    erp_index,
                    score,
                    amount_difference
                )
            )

        if not valid_candidates:
            continue

        # Select the best candidate.
        best_candidate = sorted(
            valid_candidates,
            key=lambda x: (-x[1], x[2])
        )[0]

        erp_index, score, difference = best_candidate

        # Mark the ERP record as used.
        erp.at[erp_index, "match_used"] = True

        # Mark the tool record as a duplicate.
        tool.at[
            tool_index,
            "integration_status"
        ] = "TOOL_MATCHED_DUPLICATE"

        tool.at[
            tool_index,
            "matched_record_id"
        ] = erp.at[erp_index, "record_id"]

        tool.at[
            tool_index,
            "match_method"
        ] = "NORMALIZED_PO_AMOUNT_SUPPLIER"

        tool.at[
            tool_index,
            "match_score"
        ] = round(score, 4)

    erp.drop(
        columns=["match_used"],
        inplace=True
    )

    return erp, tool


# --------------------------------------------------
# 9. Merge Both Data Sources
# --------------------------------------------------

def merge_sources(
    erp_df,
    tool_df,
    amount_tolerance=0.03,
    absolute_tolerance=1.0,
    supplier_threshold=0.90
):
    """
    Complete Company A integration pipeline.

    Returns:
        combined_df

    Every original ERP and tool record is preserved.
    """

    print("\nStarting source integration...")

    # Prepare ERP records
    erp = prepare_erp_data(erp_df)

    # Prepare invoice tool records
    tool = prepare_tool_data(tool_df)

    # Detect repeated invoice exports
    tool = detect_tool_duplicates(tool)

    # Match ERP and tool records
    erp, tool = match_erp_tool_records(
        erp,
        tool,
        amount_tolerance=amount_tolerance,
        absolute_tolerance=absolute_tolerance,
        supplier_threshold=supplier_threshold
    )

    # Combine both systems without deleting rows.
    combined = pd.concat(
        [erp, tool],
        ignore_index=True,
        sort=False
    )

    # Add final duplicate flag.
    combined["is_cross_system_duplicate"] = (
        combined["integration_status"]
        == "TOOL_MATCHED_DUPLICATE"
    )

    combined["is_tool_export_duplicate"] = (
        combined["integration_status"]
        == "TOOL_EXPORT_DUPLICATE"
    )

    print("\nSource integration completed.")

    print(
        f"ERP records: {len(erp):,}"
    )

    print(
        f"Invoice tool records: {len(tool):,}"
    )

    print(
        f"Combined records: {len(combined):,}"
    )

    print("\nIntegration status:")

    print(
        combined["integration_status"]
        .value_counts()
        .to_string()
    )

    return combined


# --------------------------------------------------
# 10. Integration Summary
# --------------------------------------------------

def generate_integration_report(combined_df):
    """
    Generate integration statistics.
    """

    report = (
        combined_df["integration_status"]
        .value_counts()
        .rename_axis("integration_status")
        .reset_index(name="record_count")
    )

    report["percentage"] = (
        report["record_count"]
        / len(combined_df)
        * 100
    ).round(2)

    return report
