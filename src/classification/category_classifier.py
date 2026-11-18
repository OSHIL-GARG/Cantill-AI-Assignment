
"""Build taxonomy classifiers and generate category predictions."""

from __future__ import annotations

import pandas as pd

from ..classifier import TaxonomyClassifier
from .taxonomy_loader import get_taxonomy_levels
from .confidence import normalize_confidence


def build_classifier(
    taxonomy: pd.DataFrame,
    training_texts=None,
    training_labels=None,
) -> TaxonomyClassifier:
    """
    Initialize the taxonomy classifier.

    If training examples are supplied, fit the supervised model.
    Otherwise, the classifier uses taxonomy text similarity.
    """

    if not isinstance(taxonomy, pd.DataFrame):
        raise TypeError("taxonomy must be a pandas DataFrame.")

    if taxonomy.empty:
        raise ValueError("Cannot build a classifier from an empty taxonomy.")

    levels = get_taxonomy_levels(taxonomy)

    if not levels:
        raise ValueError("No category hierarchy found in taxonomy.")

    model = TaxonomyClassifier(taxonomy.copy())

    if training_texts is not None or training_labels is not None:
        if training_texts is None or training_labels is None:
            raise ValueError(
                "Both training_texts and training_labels are required."
            )

        training_texts = list(training_texts)
        training_labels = list(training_labels)

        if len(training_texts) != len(training_labels):
            raise ValueError(
                "Training texts and labels must have the same length."
            )

        model.fit(training_texts, training_labels)

    return model


def predict_categories(
    model: TaxonomyClassifier,
    texts,
    taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict category paths for input text.

    Output:
        category_l1 ... category_l4
        confidence

    Unavailable taxonomy levels are returned as empty strings.
    """

    if model is None:
        raise ValueError("A classifier model is required.")

    if not isinstance(taxonomy, pd.DataFrame) or taxonomy.empty:
        raise ValueError("A valid, non-empty taxonomy is required.")

    levels = get_taxonomy_levels(taxonomy)

    if isinstance(texts, pd.Series):
        input_texts = texts.fillna("").astype(str).tolist()
    elif isinstance(texts, str):
        input_texts = [texts]
    else:
        input_texts = [
            "" if pd.isna(value) else str(value)
            for value in texts
        ]

    # Call the existing project classifier.
    predicted_paths, raw_confidence = model.predict(input_texts)

    if len(predicted_paths) != len(input_texts):
        raise RuntimeError(
            "Classifier returned an unexpected number of category paths."
        )

    if len(raw_confidence) != len(input_texts):
        raise RuntimeError(
            "Classifier returned an unexpected number of confidence scores."
        )

    confidence_values = normalize_confidence(raw_confidence)

    rows = []

    for path, confidence in zip(
        predicted_paths,
        confidence_values,
    ):
        path = path or []

        row = {}

        for index, level in enumerate(levels):
            value = path[index] if index < len(path) else ""
            row[level] = str(value).strip() if value else ""

        # The submission schema supports up to four levels.
        for level in (
            "category_l1",
            "category_l2",
            "category_l3",
            "category_l4",
        ):
            row.setdefault(level, "")

        row["confidence"] = confidence
        rows.append(row)

    columns = [
        "category_l1",
        "category_l2",
        "category_l3",
        "category_l4",
        "confidence",
    ]

    return pd.DataFrame(rows, columns=columns)


def classify_dataframe(
    df: pd.DataFrame,
    model: TaxonomyClassifier,
    taxonomy: pd.DataFrame,
    text_column: str = "classification_text",
) -> pd.DataFrame:
    """Append category predictions to the original DataFrame."""

    if text_column not in df.columns:
        raise KeyError(
            f"Classification text column not found: {text_column}"
        )

    result = df.copy()

    predictions = predict_categories(
        model=model,
        texts=result[text_column],
        taxonomy=taxonomy,
    )

    predictions.index = result.index

    for column in predictions.columns:
        result[column] = predictions[column]

    return result
