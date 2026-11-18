
"""
Module: item_matching.py

Purpose:
    Identify and group records representing the same item.

Features:
    - Material number normalization.
    - Description normalization.
    - Exact item matching.
    - Fuzzy description matching.
    - Unit and category validation.
    - Stable item key generation.
    - Item matching audit report.

Important:
    Different item codes are not automatically duplicates.
    Ambiguous matches are kept separate.
"""

import re
import hashlib
import unicodedata

import pandas as pd

from difflib import SequenceMatcher


# --------------------------------------------------
# 1. Normalize Material Number
# --------------------------------------------------

def normalize_material_number(value):
    """Normalize material numbers."""

    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))

    text = text.upper().strip()

    return re.sub(r"[^A-Z0-9]", "", text)


# --------------------------------------------------
# 2. Normalize Item Description
# --------------------------------------------------

def normalize_item_description(value):
    """
    Normalize item descriptions.

    Preserves multilingual letters and numbers.
    """

    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))

    text = text.lower().strip()

    text = re.sub(
        r"[^\w]+",
        " ",
        text
    )

    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------
# 3. Calculate Item Similarity
# --------------------------------------------------

def item_similarity(description_a, description_b):
    """Calculate item description similarity."""

    a = normalize_item_description(description_a)
    b = normalize_item_description(description_b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------
# 4. Generate Stable Item Key
# --------------------------------------------------

def generate_item_key(identity):
    """Generate a deterministic item key."""

    if not identity:
        return ""

    digest = hashlib.sha256(
        str(identity).encode("utf-8")
    ).hexdigest()[:12]

    return f"ITEM-{digest.upper()}"


# --------------------------------------------------
# 5. Prepare Item Master
# --------------------------------------------------

def build_item_master(
    df,
    item_id_column="material_number",
    description_column="description"
):
    """
    Build unique item identities.
    """

    required = [
        item_id_column,
        description_column
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing item columns: {missing}"
        )

    items = df[required].copy()

    items["item_id_normalized"] = (
        items[item_id_column].apply(
            normalize_material_number
        )
    )

    items["description_normalized"] = (
        items[description_column].apply(
            normalize_item_description
        )
    )

    items = items.drop_duplicates(
        subset=[
            "item_id_normalized",
            "description_normalized"
        ]
    ).reset_index(drop=True)

    return items


# --------------------------------------------------
# 6. Resolve Item Entities
# --------------------------------------------------

def match_items(
    df,
    item_id_column="material_number",
    description_column="description",
    unit_column=None,
    category_column=None,
    similarity_threshold=0.94
):
    """
    Resolve item identities.

    Matching priority:
        1. Exact material number.
        2. Exact normalized description.
        3. High-confidence fuzzy description matching.

    Optional evidence:
        - Unit of measurement.
        - Existing category.

    Returns:
        resolved_df
        item_master
        item_match_report
    """

    required = [
        item_id_column,
        description_column
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing item columns: {missing}"
        )

    result = df.copy()

    if "record_id" not in result.columns:
        result["record_id"] = result.index.astype(str)

    result["item_id_normalized"] = (
        result[item_id_column].apply(
            normalize_material_number
        )
    )

    result["description_normalized"] = (
        result[description_column].apply(
            normalize_item_description
        )
    )

    # Optional metadata.
    if unit_column and unit_column in result.columns:
        result["item_unit_normalized"] = (
            result[unit_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        result["item_unit_normalized"] = ""

    if category_column and category_column in result.columns:
        result["item_category_normalized"] = (
            result[category_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        result["item_category_normalized"] = ""

    # Build unique observed item identities.
    identity_columns = [
        "item_id_normalized",
        "description_normalized",
        "item_unit_normalized",
        "item_category_normalized"
    ]

    items = (
        result[identity_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    items["item_key"] = ""
    items["matching_method"] = ""
    items["matching_confidence"] = 0.0

    # First pass: exact material number.
    for idx, row in items.iterrows():

        material_id = row["item_id_normalized"]
        description = row["description_normalized"]

        if material_id:

            items.at[idx, "item_key"] = generate_item_key(
                f"MATERIAL:{material_id}"
            )

            items.at[idx, "matching_method"] = "EXACT_MATERIAL_ID"

            items.at[idx, "matching_confidence"] = 1.0

        elif description:

            items.at[idx, "item_key"] = generate_item_key(
                f"DESCRIPTION:{description}"
            )

            items.at[idx, "matching_method"] = "EXACT_DESCRIPTION"

            items.at[idx, "matching_confidence"] = 1.0

        else:

            # Missing identity should never be grouped
            # with other missing-identity records.
            items.at[idx, "item_key"] = generate_item_key(
                f"UNKNOWN:{idx}"
            )

            items.at[idx, "matching_method"] = "UNKNOWN_IDENTITY"

            items.at[idx, "matching_confidence"] = 0.0

    # Second pass: exact descriptions.
    # If the material number differs but the full normalized
    # description and metadata match, use one description key.
    described = items[
        items["description_normalized"].ne("")
    ]

    for _, group in described.groupby(
        [
            "description_normalized",
            "item_unit_normalized",
            "item_category_normalized"
        ]
    ):

        if len(group) < 2:
            continue

        key = generate_item_key(
            "DESCRIPTION:"
            + str(group.iloc[0]["description_normalized"])
            + "|UNIT:"
            + str(group.iloc[0]["item_unit_normalized"])
            + "|CATEGORY:"
            + str(group.iloc[0]["item_category_normalized"])
        )

        indices = group.index

        items.loc[indices, "item_key"] = key

        items.loc[
            indices,
            "matching_method"
        ] = "EXACT_DESCRIPTION_METADATA"

        items.loc[
            indices,
            "matching_confidence"
        ] = 1.0

    # Third pass: conservative fuzzy description matching.
    candidates = items[
        items["item_id_normalized"].eq("")
        & items["description_normalized"].ne("")
    ]

    candidate_indices = candidates.index.tolist()

    for position, idx_a in enumerate(candidate_indices):

        if items.at[idx_a, "matching_method"] == "FUZZY_DESCRIPTION":
            continue

        desc_a = items.at[idx_a, "description_normalized"]

        for idx_b in candidate_indices[position + 1:]:

            desc_b = items.at[idx_b, "description_normalized"]

            # Don't merge records with incompatible units.
            unit_a = items.at[idx_a, "item_unit_normalized"]
            unit_b = items.at[idx_b, "item_unit_normalized"]

            if unit_a and unit_b and unit_a != unit_b:
                continue

            # Don't merge records with incompatible categories.
            category_a = items.at[idx_a, "item_category_normalized"]
            category_b = items.at[idx_b, "item_category_normalized"]

            if (
                category_a
                and category_b
                and category_a != category_b
            ):
                continue

            score = item_similarity(desc_a, desc_b)

            if score >= similarity_threshold:

                # Only merge when one side is a sufficiently
                # distinctive description of the other.
                tokens_a = set(desc_a.split())
                tokens_b = set(desc_b.split())

                token_coverage = (
                    len(tokens_a.intersection(tokens_b))
                    / max(len(tokens_a), len(tokens_b))
                )

                if token_coverage < 0.80:
                    continue

                # Keep fuzzy groups conservative: a candidate
                # must match the existing canonical description.
                canonical_idx = min(idx_a, idx_b)

                canonical_desc = items.at[
                    canonical_idx,
                    "description_normalized"
                ]

                other_idx = max(idx_a, idx_b)

                if item_similarity(
                    canonical_desc,
                    items.at[other_idx, "description_normalized"]
                ) < similarity_threshold:
                    continue

                canonical_unit = items.at[
                    canonical_idx,
                    "item_unit_normalized"
                ]

                canonical_category = items.at[
                    canonical_idx,
                    "item_category_normalized"
                ]

                canonical_key = generate_item_key(
                    f"FUZZY:{canonical_desc}"
                    f"|UNIT:{canonical_unit}"
                    f"|CATEGORY:{canonical_category}"
                )

                # Do not overwrite exact material-number identities.
                if (
                    not items.at[idx_a, "item_id_normalized"]
                    and not items.at[idx_b, "item_id_normalized"]
                ):

                    items.at[idx_a, "item_key"] = canonical_key
                    items.at[idx_b, "item_key"] = canonical_key

                    items.at[idx_a, "matching_method"] = "FUZZY_DESCRIPTION"
                    items.at[idx_b, "matching_method"] = "FUZZY_DESCRIPTION"

                    items.at[idx_a, "matching_confidence"] = score
                    items.at[idx_b, "matching_confidence"] = score

    # Attach item keys to original records.
    result = result.merge(
        items,
        on=identity_columns,
        how="left",
        validate="many_to_one"
    )

    item_master = items.drop_duplicates(
        subset=["item_key"]
    ).reset_index(drop=True)

    report = (
        items.groupby("matching_method")
        .agg(
            unique_item_keys=("item_key", "nunique"),
            observed_item_identities=("item_key", "size")
        )
        .reset_index()
    )

    return result, item_master, report
