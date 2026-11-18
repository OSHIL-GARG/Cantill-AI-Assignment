
"""
Module: purchase_matching.py

Purpose:
    Match duplicate purchase transactions across source systems.

Features:
    - Normalize PO and invoice references.
    - Exact PO matching.
    - Invoice number matching.
    - Supplier and amount comparison.
    - Date proximity matching.
    - Fallback matching when PO is missing.
    - One-to-one matching.
    - Matching audit report.

Returns:
    Original records with matching metadata.
"""

import re
import pandas as pd
import numpy as np

from difflib import SequenceMatcher


def normalize_reference(value):
    """Normalize PO or invoice references."""

    if pd.isna(value):
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).upper()
    )


def normalize_name(value):
    """Normalize supplier names."""

    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def similarity(a, b):
    """Calculate string similarity between 0 and 1."""

    a = normalize_name(a)
    b = normalize_name(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def amount_similarity(a, b, tolerance=0.03):
    """
    Compare two purchase amounts.

    Returns:
        1.0 for an exact match.
        A score between 0 and 1 for acceptable differences.
        0.0 for unacceptable differences.
    """

    if pd.isna(a) or pd.isna(b):
        return 0.0

    a = float(a)
    b = float(b)

    difference = abs(a - b)

    allowed_difference = max(
        abs(a) * tolerance,
        1.0
    )

    if difference > allowed_difference:
        return 0.0

    return max(
        0.0,
        1.0 - difference / allowed_difference
    )


def date_similarity(date_a, date_b, max_days=30):
    """Compare transaction dates."""

    if pd.isna(date_a) or pd.isna(date_b):
        return 0.0

    days = abs(
        (pd.Timestamp(date_a) - pd.Timestamp(date_b)).days
    )

    if days > max_days:
        return 0.0

    return max(0.0, 1.0 - days / max_days)


def match_purchase_records(
    df,
    source_column="source_type",
    po_column="po_normalized",
    invoice_column="invoice_number",
    supplier_column="supplier_key",
    amount_column="net_amount",
    date_column="invoice_date",
    amount_tolerance=0.03,
    supplier_threshold=0.85,
    max_date_days=30,
    minimum_score=0.65,
    fallback_without_po=True
):
    """
    Identify matching purchases between source systems.

    Matching priorities:
        1. Exact normalized PO + amount.
        2. Exact invoice reference + supplier + amount.
        3. Missing PO fallback using supplier, amount and date.

    Each record can participate in only one match.

    Returns:
        matched_df:
            Original DataFrame with matching metadata.

        matches_df:
            One row per matched pair.
    """

    required = [
        source_column,
        supplier_column,
        amount_column
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Required purchase matching columns missing: {missing}"
        )

    result = df.copy().reset_index(drop=True)

    if po_column not in result.columns:
        result[po_column] = ""

    if invoice_column not in result.columns:
        result[invoice_column] = ""

    if date_column not in result.columns:
        result[date_column] = pd.NaT

    result[po_column] = result[po_column].map(normalize_reference)
    result[invoice_column] = result[invoice_column].map(normalize_reference)

    result["purchase_match_id"] = ""
    result["matched_record_id"] = ""
    result["purchase_match_method"] = ""
    result["purchase_match_score"] = 0.0
    result["is_purchase_duplicate"] = False

    if "record_id" not in result.columns:
        result["record_id"] = result.index.astype(str)

    # Create candidate pairs using indexed reference groups.
    po_groups = {}
    invoice_groups = {}

    for idx, row in result.iterrows():

        po = row[po_column]
        invoice = row[invoice_column]

        if po:
            po_groups.setdefault(po, []).append(idx)

        if invoice:
            invoice_groups.setdefault(invoice, []).append(idx)

    candidate_pairs = set()

    def add_pairs(groups):
        for indices in groups.values():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = indices[i], indices[j]

                    if result.at[a, source_column] != result.at[b, source_column]:
                        candidate_pairs.add((a, b))

    add_pairs(po_groups)
    add_pairs(invoice_groups)

    # Fallback blocking for records without PO references.
    if fallback_without_po:

        fallback_groups = {}

        for idx, row in result.iterrows():

            if row[po_column]:
                continue

            supplier = str(row[supplier_column])
            amount = row[amount_column]

            if pd.isna(amount) or not supplier or supplier == "nan":
                continue

            # Amount bucket limits the number of candidate comparisons.
            amount_bucket = round(float(amount), -1)

            key = (supplier, amount_bucket)

            fallback_groups.setdefault(key, []).append(idx)

        add_pairs(fallback_groups)

    scored_candidates = []

    for idx_a, idx_b in candidate_pairs:

        a = result.loc[idx_a]
        b = result.loc[idx_b]

        # Do not compare records from the same source.
        if a[source_column] == b[source_column]:
            continue

        amount_score = amount_similarity(
            a[amount_column],
            b[amount_column],
            tolerance=amount_tolerance
        )

        if amount_score == 0:
            continue

        supplier_score = similarity(
            a[supplier_column],
            b[supplier_column]
        )

        # Stable supplier IDs can match exactly.
        supplier_exact = (
            str(a[supplier_column]) == str(b[supplier_column])
        )

        if not supplier_exact and supplier_score < supplier_threshold:
            continue

        po_exact = bool(a[po_column]) and a[po_column] == b[po_column]

        invoice_exact = (
            bool(a[invoice_column])
            and a[invoice_column] == b[invoice_column]
        )

        date_score = date_similarity(
            a[date_column],
            b[date_column],
            max_days=max_date_days
        )

        if po_exact:
            score = (
                0.50 * amount_score
                + 0.25 * float(supplier_exact or supplier_score >= supplier_threshold)
                + 0.15 * date_score
                + 0.10
            )
            method = "PO_AMOUNT_SUPPLIER"

        elif invoice_exact:
            score = (
                0.50 * amount_score
                + 0.30 * float(supplier_exact or supplier_score >= supplier_threshold)
                + 0.20 * date_score
            )
            method = "INVOICE_AMOUNT_SUPPLIER"

        else:
            score = (
                0.55 * amount_score
                + 0.30 * supplier_score
                + 0.15 * date_score
            )
            method = "SUPPLIER_AMOUNT_DATE_FALLBACK"

        if score >= minimum_score:
            scored_candidates.append(
                (score, idx_a, idx_b, method)
            )

    # Greedy one-to-one assignment, highest score first.
    scored_candidates.sort(reverse=True)

    used_indices = set()
    matches = []

    for score, idx_a, idx_b, method in scored_candidates:

        if idx_a in used_indices or idx_b in used_indices:
            continue

        used_indices.update([idx_a, idx_b])

        record_a = result.at[idx_a, "record_id"]
        record_b = result.at[idx_b, "record_id"]

        # ERP is preferred as the primary record.
        source_a = str(result.at[idx_a, source_column]).upper()
        source_b = str(result.at[idx_b, source_column]).upper()

        if source_a == "ERP":
            primary_idx, duplicate_idx = idx_a, idx_b
        elif source_b == "ERP":
            primary_idx, duplicate_idx = idx_b, idx_a
        else:
            primary_idx, duplicate_idx = min(idx_a, idx_b), max(idx_a, idx_b)

        primary_id = result.at[primary_idx, "record_id"]
        duplicate_id = result.at[duplicate_idx, "record_id"]

        match_id = f"PM-{len(matches) + 1:07d}"

        result.at[primary_idx, "purchase_match_id"] = match_id
        result.at[duplicate_idx, "purchase_match_id"] = match_id

        result.at[duplicate_idx, "matched_record_id"] = primary_id
        result.at[primary_idx, "matched_record_id"] = duplicate_id

        result.at[primary_idx, "purchase_match_method"] = method
        result.at[duplicate_idx, "purchase_match_method"] = method

        result.at[primary_idx, "purchase_match_score"] = round(score, 4)
        result.at[duplicate_idx, "purchase_match_score"] = round(score, 4)

        result.at[duplicate_idx, "is_purchase_duplicate"] = True

        matches.append({
            "purchase_match_id": match_id,
            "primary_record_id": primary_id,
            "duplicate_record_id": duplicate_id,
            "matching_method": method,
            "confidence": round(score, 4)
        })

    matches_df = pd.DataFrame(
        matches,
        columns=[
            "purchase_match_id",
            "primary_record_id",
            "duplicate_record_id",
            "matching_method",
            "confidence"
        ]
    )

    return result, matches_df
