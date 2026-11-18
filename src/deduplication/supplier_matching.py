
"""
Module: supplier_matching.py

Purpose:
    Identify and group records belonging to the same supplier.

Features:
    - Supplier name normalization.
    - Exact supplier ID matching.
    - Fuzzy supplier name matching.
    - Stable supplier key generation.
    - Matching audit report.

Important:
    Fuzzy matches are conservative.
    Ambiguous suppliers are not automatically merged.
"""

import re
import hashlib
import unicodedata

import pandas as pd

from difflib import SequenceMatcher


# --------------------------------------------------
# 1. Normalize Supplier Name
# --------------------------------------------------

def normalize_supplier_name(value):
    """Normalize supplier names for entity resolution."""

    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.lower().strip()

    # Remove common legal suffixes.
    text = re.sub(
        r"\b(limited|ltd|llc|incorporated|inc|corp|corporation|plc|pty)\b",
        " ",
        text
    )

    text = re.sub(r"[^\w]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------
# 2. Calculate Name Similarity
# --------------------------------------------------

def supplier_similarity(name_a, name_b):
    """Calculate supplier name similarity."""

    a = normalize_supplier_name(name_a)
    b = normalize_supplier_name(name_b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------
# 3. Generate Stable Supplier Key
# --------------------------------------------------

def generate_supplier_key(identity):
    """
    Generate a deterministic supplier key.

    The same normalized identity generates
    the same key across multiple executions.
    """

    if not identity:
        return ""

    identity = str(identity).strip().lower()

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:12]

    return f"SUP-{digest.upper()}"


# --------------------------------------------------
# 4. Build Supplier Master
# --------------------------------------------------

def build_supplier_master(
    df,
    supplier_name_column="supplier_name",
    supplier_id_column=None
):
    """
    Build unique supplier records.

    Returns one row per observed supplier identity.
    """

    if supplier_name_column not in df.columns:
        raise ValueError(
            f"Supplier name column not found: {supplier_name_column}"
        )

    columns = [supplier_name_column]

    if supplier_id_column and supplier_id_column in df.columns:
        columns.append(supplier_id_column)

    suppliers = df[columns].copy()

    suppliers["supplier_name_normalized"] = (
        suppliers[supplier_name_column].apply(
            normalize_supplier_name
        )
    )

    suppliers["supplier_id_normalized"] = (
        suppliers[supplier_id_column]
        .fillna("")
        .astype(str)
        .str.strip()
        if supplier_id_column and supplier_id_column in suppliers.columns
        else ""
    )

    suppliers = suppliers.drop_duplicates(
        subset=[
            "supplier_name_normalized",
            "supplier_id_normalized"
        ]
    ).reset_index(drop=True)

    return suppliers


# --------------------------------------------------
# 5. Match Supplier Entities
# --------------------------------------------------

def match_suppliers(
    df,
    supplier_name_column="supplier_name",
    supplier_id_column=None,
    similarity_threshold=0.92
):
    """
    Resolve supplier entities.

    Matching priority:
        1. Existing supplier ID.
        2. Exact normalized supplier name.
        3. Conservative fuzzy name matching.

    Ambiguous fuzzy matches are not merged.

    Returns:
        resolved_df
        supplier_master
        supplier_match_report
    """

    if supplier_name_column not in df.columns:
        raise ValueError(
            f"Missing supplier name column: {supplier_name_column}"
        )

    result = df.copy()

    if "record_id" not in result.columns:
        result["record_id"] = result.index.astype(str)

    result["supplier_name_normalized"] = (
        result[supplier_name_column].apply(
            normalize_supplier_name
        )
    )

    if supplier_id_column and supplier_id_column in result.columns:

        result["supplier_id_normalized"] = (
            result[supplier_id_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        result.loc[
            result["supplier_id_normalized"].str.lower().isin(
                ["nan", "none", "null"]
            ),
            "supplier_id_normalized"
        ] = ""

    else:
        result["supplier_id_normalized"] = ""

    # Build unique supplier identities.
    identities = (
        result[
            [
                "supplier_name_normalized",
                "supplier_id_normalized"
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    identities["supplier_key"] = ""
    identities["matching_method"] = ""
    identities["matching_confidence"] = 0.0

    # Stable key based on supplier ID when available.
    id_groups = {}

    for idx, row in identities.iterrows():

        supplier_id = row["supplier_id_normalized"]
        name = row["supplier_name_normalized"]

        if supplier_id:

            identity = f"ID:{supplier_id}"

            identities.at[idx, "supplier_key"] = (
                generate_supplier_key(identity)
            )

            identities.at[idx, "matching_method"] = "EXACT_SUPPLIER_ID"
            identities.at[idx, "matching_confidence"] = 1.0

            id_groups.setdefault(supplier_id, []).append(idx)

        elif name:

            # Exact normalized names can share the same key.
            identity = f"NAME:{name}"

            identities.at[idx, "supplier_key"] = (
                generate_supplier_key(identity)
            )

            identities.at[idx, "matching_method"] = "NORMALIZED_NAME"
            identities.at[idx, "matching_confidence"] = 1.0

        else:

            # Never merge suppliers with missing identity.
            identity = f"UNKNOWN:{idx}"

            identities.at[idx, "supplier_key"] = (
                generate_supplier_key(identity)
            )

            identities.at[idx, "matching_method"] = "UNKNOWN_IDENTITY"
            identities.at[idx, "matching_confidence"] = 0.0

    # Exact supplier IDs identify one entity even when
    # the supplier's name is typed differently.
    for supplier_id, indices in id_groups.items():

        if len(indices) <= 1:
            continue

        canonical_key = identities.at[
            indices[0], "supplier_key"
        ]

        identities.loc[
            indices,
            "supplier_key"
        ] = canonical_key

    # Fuzzy matching only among records without supplier IDs.
    no_id_indices = identities.index[
        identities["supplier_id_normalized"].eq("")
        & identities["supplier_name_normalized"].ne("")
    ].tolist()

    parent = {idx: idx for idx in no_id_indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)

        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    fuzzy_candidates = []

    for pos, idx_a in enumerate(no_id_indices):

        name_a = identities.at[
            idx_a, "supplier_name_normalized"
        ]

        for idx_b in no_id_indices[pos + 1:]:

            name_b = identities.at[
                idx_b, "supplier_name_normalized"
            ]

            # Require a shared leading token to reduce comparisons.
            tokens_a = set(name_a.split())
            tokens_b = set(name_b.split())

            if not tokens_a.intersection(tokens_b):
                continue

            score = supplier_similarity(name_a, name_b)

            if score >= similarity_threshold:

                fuzzy_candidates.append(
                    (score, idx_a, idx_b)
                )

    # Avoid transitive merging of weakly related suppliers.
    fuzzy_candidates.sort(reverse=True)

    cluster_members = {
        idx: {idx} for idx in no_id_indices
    }

    for score, idx_a, idx_b in fuzzy_candidates:

        root_a = find(idx_a)
        root_b = find(idx_b)

        if root_a == root_b:
            continue

        combined = (
            cluster_members[root_a]
            | cluster_members[root_b]
        )

        names = [
            identities.at[i, "supplier_name_normalized"]
            for i in combined
        ]

        # Every pair in a cluster must meet the threshold.
        all_similar = all(
            supplier_similarity(names[i], names[j])
            >= similarity_threshold
            for i in range(len(names))
            for j in range(i + 1, len(names))
        )

        if not all_similar:
            continue

        union(root_a, root_b)

        new_root = find(root_a)

        cluster_members[new_root] = combined

        if new_root != root_a:
            cluster_members.pop(root_a, None)

        if new_root != root_b:
            cluster_members.pop(root_b, None)

    for idx in no_id_indices:

        root = find(idx)

        members = cluster_members.get(root, {idx})

        if len(members) > 1:

            canonical_name = min(
                identities.loc[
                    list(members),
                    "supplier_name_normalized"
                ].tolist(),
                key=lambda x: (len(x), x)
            )

            key = generate_supplier_key(
                f"NAME:{canonical_name}"
            )

            identities.at[idx, "supplier_key"] = key
            identities.at[idx, "matching_method"] = "FUZZY_NAME"
            identities.at[idx, "matching_confidence"] = min(
                supplier_similarity(
                    identities.at[idx, "supplier_name_normalized"],
                    identities.at[other, "supplier_name_normalized"]
                )
                for other in members
            )

    # Attach resolved keys to original records.
    result = result.merge(
        identities[
            [
                "supplier_name_normalized",
                "supplier_id_normalized",
                "supplier_key",
                "matching_method",
                "matching_confidence"
            ]
        ],
        on=[
            "supplier_name_normalized",
            "supplier_id_normalized"
        ],
        how="left",
        validate="many_to_one"
    )

    master = identities.drop_duplicates(
        subset=["supplier_key"]
    ).reset_index(drop=True)

    report = (
        identities.groupby("matching_method")
        .agg(
            supplier_groups=("supplier_key", "nunique"),
            observed_identities=("supplier_key", "size")
        )
        .reset_index()
    )

    return result, master, report
