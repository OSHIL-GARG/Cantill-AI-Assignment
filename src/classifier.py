
"""
Taxonomy classification module for the Cantill Spend Intelligence project.

Supports:
1. Zero-shot taxonomy classification using TF-IDF and cosine similarity.
2. Supervised classification using TF-IDF and Logistic Regression.
3. Mapping predicted categories to the taxonomy hierarchy.
4. Confidence scores between 0 and 1.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


class TaxonomyClassifier:
    """Classify procurement descriptions into taxonomy categories."""

    TAXONOMY_ALIASES = {
        "l1": ["category_l1", "l1", "level_1", "level1"],
        "l2": ["category_l2", "l2", "level_2", "level2"],
        "l3": ["category_l3", "l3", "level_3", "level3"],
        "l4": ["category_l4", "l4", "level_4", "level4"],
    }

    def __init__(self, taxonomy: pd.DataFrame):
        """
        Initialize classifier with a taxonomy DataFrame.

        Expected taxonomy columns may include:
            category_l1, category_l2, category_l3, category_l4

        Short alternatives such as l1, l2, l3, and l4 are also supported.
        """
        if not isinstance(taxonomy, pd.DataFrame):
            raise TypeError("taxonomy must be a pandas DataFrame.")

        if taxonomy.empty:
            raise ValueError("Taxonomy DataFrame is empty.")

        self.taxonomy = taxonomy.copy()
        self.taxonomy.columns = [
            str(column).strip().lower()
            for column in self.taxonomy.columns
        ]

        self.level_columns = self._find_level_columns()

        if not self.level_columns:
            raise ValueError(
                "No taxonomy hierarchy columns found. "
                "Expected columns such as category_l1, category_l2, "
                "category_l3, category_l4 or l1, l2, l3, l4."
            )

        self.taxonomy_paths = self._build_taxonomy_paths()

        if not self.taxonomy_paths:
            raise ValueError(
                "Taxonomy does not contain any valid category paths."
            )

        self.taxonomy_texts = [
            " ".join(path)
            for path in self.taxonomy_paths
        ]

        self.taxonomy_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )

        self.taxonomy_matrix = None

        self.model = None
        self.training_vectorizer = None
        self.label_to_path = {}
        self.is_trained = False

        self._fit_taxonomy_vectors()

    # ---------------------------------------------------------
    # Taxonomy preparation
    # ---------------------------------------------------------

    def _find_level_columns(self) -> dict[int, str]:
        """Find the available taxonomy hierarchy columns."""
        result = {}

        for level, aliases in self.TAXONOMY_ALIASES.items():
            for alias in aliases:
                if alias in self.taxonomy.columns:
                    number = int(level[1:])
                    result[number] = alias
                    break

        return result

    @staticmethod
    def _clean_value(value) -> str:
        """Convert a taxonomy value into clean text."""
        if pd.isna(value):
            return ""

        value = str(value).strip()

        if value.lower() in {"nan", "none", "null", "n/a"}:
            return ""

        return re.sub(r"\s+", " ", value)

    def _build_taxonomy_paths(self) -> list[list[str]]:
        """Create unique taxonomy paths from the taxonomy DataFrame."""
        paths = []
        seen = set()

        for _, row in self.taxonomy.iterrows():
            path = []

            for level in range(1, 5):
                column = self.level_columns.get(level)

                if column:
                    value = self._clean_value(row[column])

                    if value:
                        path.append(value)

            if not path:
                continue

            key = tuple(value.lower() for value in path)

            if key not in seen:
                paths.append(path)
                seen.add(key)

        return paths

    def _fit_taxonomy_vectors(self) -> None:
        """Create TF-IDF vectors for taxonomy category paths."""
        try:
            self.taxonomy_matrix = (
                self.taxonomy_vectorizer.fit_transform(
                    self.taxonomy_texts
                )
            )
        except ValueError:
            # Handles taxonomies containing only blank or stop words.
            self.taxonomy_matrix = None

    # ---------------------------------------------------------
    # Supervised training
    # ---------------------------------------------------------

    def fit(
        self,
        training_texts: Iterable[str],
        training_labels: Iterable[str],
    ):
        """
        Train a supervised text classifier.

        Parameters:
            training_texts:
                Procurement descriptions.

            training_labels:
                Category labels, normally category_l3.

        Returns:
            self
        """
        texts = list(training_texts)
        labels = list(training_labels)

        if len(texts) != len(labels):
            raise ValueError(
                "Training descriptions and labels must have equal lengths."
            )

        clean_texts = []
        clean_labels = []

        for text, label in zip(texts, labels):
            text = self._clean_value(text)
            label = self._clean_value(label)

            if text and label:
                clean_texts.append(text)
                clean_labels.append(label)

        if len(clean_texts) < 2:
            raise ValueError(
                "At least two valid labelled training records are required."
            )

        unique_labels = sorted(set(clean_labels))

        # Store the corresponding taxonomy path for every training label.
        self.label_to_path = {}

        for label in unique_labels:
            matching_paths = [
                path
                for path in self.taxonomy_paths
                if path[-1].lower() == label.lower()
            ]

            if not matching_paths:
                matching_paths = [
                    path
                    for path in self.taxonomy_paths
                    if any(
                        value.lower() == label.lower()
                        for value in path
                    )
                ]

            if matching_paths:
                self.label_to_path[label] = matching_paths[0]
            else:
                # Retain the label even if it is not present in taxonomy.
                self.label_to_path[label] = [label]

        self.training_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )

        try:
            training_matrix = (
                self.training_vectorizer.fit_transform(clean_texts)
            )
        except ValueError as exc:
            raise ValueError(
                "Unable to create training text vectors. "
                "Check whether the descriptions contain meaningful words."
            ) from exc

        # Logistic Regression requires at least two classes.
        if len(unique_labels) >= 2:
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )

            self.model.fit(training_matrix, clean_labels)
            self.is_trained = True

        else:
            # Single-class training cannot fit Logistic Regression.
            self.model = None
            self.is_trained = False

        self.training_texts = clean_texts
        self.training_labels = clean_labels
        self.training_matrix = training_matrix

        return self

    # ---------------------------------------------------------
    # Supervised prediction
    # ---------------------------------------------------------

    def _supervised_predict(
        self,
        descriptions: list[str],
    ) -> tuple[list[list[str]], list[float]]:
        """Predict using the trained Logistic Regression model."""
        vectors = self.training_vectorizer.transform(descriptions)

        probabilities = self.model.predict_proba(vectors)
        predictions = self.model.predict(vectors)

        paths = []
        confidence = []

        for prediction, probability_row in zip(
            predictions,
            probabilities,
        ):
            label = str(prediction)

            path = self.label_to_path.get(label, [label])

            paths.append(path)

            confidence.append(
                float(np.max(probability_row))
            )

        return paths, confidence

    # ---------------------------------------------------------
    # Similarity-based prediction
    # ---------------------------------------------------------

    def _similarity_predict(
        self,
        descriptions: list[str],
    ) -> tuple[list[list[str]], list[float]]:
        """
        Predict using cosine similarity against taxonomy paths.

        This method is used when no supervised model is available.
        """
        if self.taxonomy_matrix is None:
            return (
                [["", "", "", ""] for _ in descriptions],
                [0.0 for _ in descriptions],
            )

        vectors = self.taxonomy_vectorizer.transform(descriptions)

        similarities = cosine_similarity(
            vectors,
            self.taxonomy_matrix,
        )

        paths = []
        confidence = []

        for similarity_row in similarities:
            if len(similarity_row) == 0:
                paths.append(["", "", "", ""])
                confidence.append(0.0)
                continue

            best_index = int(np.argmax(similarity_row))

            best_path = self.taxonomy_paths[best_index]

            score = float(similarity_row[best_index])

            paths.append(best_path)
            confidence.append(
                float(np.clip(score, 0.0, 1.0))
            )

        return paths, confidence

    # ---------------------------------------------------------
    # Public prediction method
    # ---------------------------------------------------------

    def predict(
        self,
        descriptions: Iterable[str],
    ) -> tuple[list[list[str]], list[float]]:
        """
        Predict taxonomy paths for procurement descriptions.

        Returns:
            paths:
                List of taxonomy paths.

            confidence:
                List of confidence scores between 0 and 1.

        Example:
            paths, confidence = classifier.predict(
                ["Office laptop purchase", "Building cleaning"]
            )
        """
        descriptions = [
            self._clean_value(description)
            for description in descriptions
        ]

        if not descriptions:
            return [], []

        # Use supervised prediction when the model is trained.
        if self.is_trained and self.model is not None:
            paths, confidence = self._supervised_predict(
                descriptions
            )
        else:
            paths, confidence = self._similarity_predict(
                descriptions
            )

        # Ensure exactly four hierarchy positions.
        normalized_paths = []

        for path in paths:
            path = list(path[:4])
            path.extend([""] * (4 - len(path)))
            normalized_paths.append(path)

        return normalized_paths, confidence

    def predict_one(
        self,
        description: str,
    ) -> tuple[list[str], float]:
        """Predict a single procurement description."""
        paths, confidence = self.predict([description])

        return paths[0], confidence[0]


# -------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------

if __name__ == "__main__":
    sample_taxonomy = pd.DataFrame({
        "category_l1": [
            "IT",
            "IT",
            "Facilities",
        ],
        "category_l2": [
            "Hardware",
            "Software",
            "Maintenance",
        ],
        "category_l3": [
            "Computers",
            "Applications",
            "Cleaning",
        ],
        "category_l4": [
            "Laptops",
            "Business Software",
            "Office Cleaning",
        ],
    })

    classifier = TaxonomyClassifier(sample_taxonomy)

    descriptions = [
        "Purchase of new laptop computers",
        "Office cleaning services",
    ]

    paths, scores = classifier.predict(descriptions)

    for description, path, score in zip(
        descriptions,
        paths,
        scores,
    ):
        print(f"\nDescription: {description}")
        print(f"Predicted path: {path}")
        print(f"Confidence: {score:.4f}")
