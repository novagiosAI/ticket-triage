"""The classifier: character n-grams plus a linear model.

Why not a transformer. Three constraints decide this:

* **It has to run without a GPU**, on whatever server the client already has.
* **It has to handle code-switching.** A Moroccan ticket mixes Darija in Arabic
  script, Darija in Latin letters, French and English — sometimes in one
  sentence. Word-level features fragment across those; character n-grams do
  not, because they see ``imprim`` in *imprimante* and ``طابع`` in ``الطابعة``
  as ordinary substrings either way.
* **It has to be honest about its training data.** The data is synthetic
  (see :mod:`ticket_triage.data`), which bounds what any model can learn from
  it. Spending a transformer's compute on template text buys accuracy on the
  templates, not on real tickets.

A word-level vectorizer runs alongside the character one, so ordinary word
evidence still counts where it exists. Swapping in a transformer later means
replacing this module and nothing else — the CLI, evaluation and demo talk to
:class:`TicketTriageModel` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

#: Bumped whenever the feature pipeline changes shape, so an old artifact is
#: refused instead of silently producing nonsense.
ARTIFACT_VERSION = 1


def _build_features() -> FeatureUnion:
    """Character n-grams plus words, both TF-IDF weighted."""
    return FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=200_000,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def _lowercase(texts: list[str]) -> list[str]:
    return [text.lower() for text in texts]


def build_pipeline(class_weight: str | None = None) -> Pipeline:
    """The full text-to-label pipeline.

    ``class_weight`` is a real trade-off, measured on this data:

    ==============  ========  ==========
    class_weight    accuracy  macro-F1
    ==============  ========  ==========
    ``None``        0.497     0.355
    ``"balanced"``  0.384     0.385
    ==============  ========  ==========

    ``None`` is the default because it maximises accuracy, while
    ``"balanced"`` trades accuracy for recall on rare priorities. Pass
    ``"balanced"`` when missing a rare high-priority ticket costs more than a
    wrong guess on a common one. The figures above predate the severity
    phrasing added to the generator; rerun ``ticket-triage train`` for current
    numbers.
    """
    return Pipeline(
        [
            ("lower", FunctionTransformer(_lowercase)),
            ("features", _build_features()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=4.0,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


@dataclass
class TicketTriageModel:
    """Predicts a ticket's category and priority from its text.

    Two independent classifiers share one interface, because the two labels are
    not the same problem: category is written in the words, while priority
    depends on urgency and impact that the text only hints at.
    """

    category: Pipeline
    priority: Pipeline
    version: int = ARTIFACT_VERSION
    metadata: dict[str, Any] | None = None

    @classmethod
    def train(
        cls,
        texts: list[str],
        categories: list[str],
        priorities: list[int],
        *,
        metadata: dict[str, Any] | None = None,
        class_weight: str | None = None,
    ) -> TicketTriageModel:
        if not texts:
            raise ValueError("no training data")
        if not len(texts) == len(categories) == len(priorities):
            raise ValueError("texts, categories and priorities must be the same length")

        category_model = build_pipeline(class_weight=class_weight)
        category_model.fit(texts, categories)

        priority_model = build_pipeline(class_weight=class_weight)
        priority_model.fit(texts, [str(value) for value in priorities])

        return cls(
            category=category_model,
            priority=priority_model,
            metadata=metadata or {},
        )

    def predict(self, text: str) -> dict[str, Any]:
        """Predict both labels for one ticket, with confidences."""
        category_probabilities = self.category.predict_proba([text])[0]
        priority_probabilities = self.priority.predict_proba([text])[0]

        category_index = int(category_probabilities.argmax())
        priority_index = int(priority_probabilities.argmax())

        return {
            "category": str(self.category.classes_[category_index]),
            "category_confidence": float(category_probabilities[category_index]),
            "priority": int(self.priority.classes_[priority_index]),
            "priority_confidence": float(priority_probabilities[priority_index]),
        }

    def predict_many(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self.predict(text) for text in texts]

    def top_categories(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """The k most likely categories, so a human can see the runners-up."""
        probabilities = self.category.predict_proba([text])[0]
        ranked = sorted(
            zip(self.category.classes_, probabilities, strict=True),
            key=lambda pair: -pair[1],
        )
        return [(str(label), float(score)) for label, score in ranked[:k]]

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": self.version,
                "category": self.category,
                "priority": self.priority,
                "metadata": self.metadata,
            },
            destination,
            compress=3,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> TicketTriageModel:
        payload = joblib.load(Path(path))
        version = payload.get("version")
        if version != ARTIFACT_VERSION:
            raise ValueError(
                f"artifact version {version} does not match {ARTIFACT_VERSION}; "
                "retrain with the current code"
            )
        return cls(
            category=payload["category"],
            priority=payload["priority"],
            version=version,
            metadata=payload.get("metadata") or {},
        )
