"""Evaluation, with the baselines that make the numbers mean something.

An accuracy figure on its own is not evidence. Three comparisons are reported
alongside it, because each one can expose a model that looks good and is not:

* **Majority class** — always predict the commonest label. If the model barely
  beats this, it has learned nothing.
* **Category only** — for priority, the accuracy reachable from the topic
  alone. Beating it is what separates "recognises the word VPN" from "read that
  the whole site is blocked".
* **Per-language** — accuracy on French, Arabic and Darija separately. A single
  aggregate can hide a model that handles French well and Darija badly, which
  for this use case is the failure that matters most.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, f1_score

from .data import Dataset
from .model import TicketTriageModel


@dataclass(frozen=True, slots=True)
class TaskScore:
    """Scores for one label, against its baseline."""

    accuracy: float
    macro_f1: float
    majority_baseline: float

    @property
    def lift_over_majority(self) -> float:
        """Accuracy points gained over always predicting the commonest label."""
        return self.accuracy - self.majority_baseline


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Everything needed to judge the model."""

    category: TaskScore
    priority: TaskScore
    by_language: dict[str, dict[str, float]] = field(default_factory=dict)
    category_report: str = ""
    priority_report: str = ""
    test_size: int = 0
    #: Priority accuracy reachable from the category alone. ``None`` when no
    #: training set was supplied to compute it.
    category_only_priority: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_size": self.test_size,
            "category": {
                "accuracy": round(self.category.accuracy, 4),
                "macro_f1": round(self.category.macro_f1, 4),
                "majority_baseline": round(self.category.majority_baseline, 4),
                "lift": round(self.category.lift_over_majority, 4),
            },
            "priority": {
                "accuracy": round(self.priority.accuracy, 4),
                "macro_f1": round(self.priority.macro_f1, 4),
                "majority_baseline": round(self.priority.majority_baseline, 4),
                "lift": round(self.priority.lift_over_majority, 4),
            },
            "category_only_priority": (
                round(self.category_only_priority, 4)
                if self.category_only_priority is not None
                else None
            ),
            "by_language": {
                language: {key: round(value, 4) for key, value in scores.items()}
                for language, scores in self.by_language.items()
            },
        }


def category_only_baseline(train: Dataset, test: Dataset) -> float:
    """Priority accuracy reachable from the **category alone**, about 0.51 here.

    This is the baseline that actually tests whether the model reads severity.
    Topic is easy: "VPN" tickets skew high-priority, "printer" tickets skew low.
    A model that only recognises the topic lands near this number. A model that
    also reads *how many people are affected* and *how soon it matters* goes
    well past it.

    It is a baseline, not a ceiling — the text carries severity phrasing on top
    of the topic, so beating it is both possible and the point. An earlier
    version of the generator omitted that phrasing, and a model trained on it
    scored 0.47 against a 0.51 majority baseline: worse than guessing, because
    there was nothing beyond the topic to learn.

    Computed on ``train`` and applied to ``test``, the way a real predictor
    would have to.
    """
    if not len(train) or not len(test):
        return 0.0

    per_category: dict[str, Counter[int]] = {}
    for category, priority in zip(train.categories, train.priorities, strict=True):
        per_category.setdefault(category, Counter())[priority] += 1

    fallback = Counter(train.priorities).most_common(1)[0][0]
    best = {
        category: counts.most_common(1)[0][0]
        for category, counts in per_category.items()
    }

    correct = sum(
        best.get(category, fallback) == priority
        for category, priority in zip(test.categories, test.priorities, strict=True)
    )
    return correct / len(test)


def majority_baseline(labels: list[Any]) -> float:
    """Accuracy of always predicting the commonest label."""
    if not labels:
        return 0.0
    most_common = Counter(str(label) for label in labels).most_common(1)[0][1]
    return most_common / len(labels)


def _score(
    true: list[str], predicted: list[str], baseline_labels: list[Any]
) -> TaskScore:
    return TaskScore(
        accuracy=float(accuracy_score(true, predicted)),
        macro_f1=float(f1_score(true, predicted, average="macro", zero_division=0)),
        majority_baseline=majority_baseline(baseline_labels),
    )


def evaluate(
    model: TicketTriageModel,
    test: Dataset,
    train: Dataset | None = None,
) -> Evaluation:
    """Score ``model`` on ``test``, overall and per language.

    Pass ``train`` to also report the priority information ceiling, which is
    what makes the priority number interpretable.
    """
    if not len(test):
        raise ValueError("empty test set")

    predicted_categories = [str(label) for label in model.category.predict(test.texts)]
    predicted_priorities = [str(label) for label in model.priority.predict(test.texts)]
    true_priorities = [str(value) for value in test.priorities]

    by_language: dict[str, dict[str, float]] = {}
    for language in sorted(set(test.languages)):
        part = test.subset(language)
        if not len(part):
            continue
        part_categories = [str(x) for x in model.category.predict(part.texts)]
        part_priorities = [str(x) for x in model.priority.predict(part.texts)]
        by_language[language] = {
            "size": float(len(part)),
            "category_accuracy": float(
                accuracy_score(part.categories, part_categories)
            ),
            "priority_accuracy": float(
                accuracy_score([str(v) for v in part.priorities], part_priorities)
            ),
        }

    return Evaluation(
        category=_score(test.categories, predicted_categories, test.categories),
        priority=_score(true_priorities, predicted_priorities, test.priorities),
        by_language=by_language,
        category_report=classification_report(
            test.categories, predicted_categories, zero_division=0
        ),
        priority_report=classification_report(
            true_priorities, predicted_priorities, zero_division=0
        ),
        test_size=len(test),
        category_only_priority=(
            category_only_baseline(train, test) if train is not None else None
        ),
    )


def format_evaluation(evaluation: Evaluation) -> str:
    """A readable summary for the terminal and the model card."""
    lines = [
        f"Test set: {evaluation.test_size} tickets",
        "",
        f"{'task':<10} {'accuracy':>9} {'macro-F1':>9} {'majority':>9} {'lift':>8}",
        f"{'-' * 48}",
    ]
    for name, score in (
        ("category", evaluation.category),
        ("priority", evaluation.priority),
    ):
        lines.append(
            f"{name:<10} {score.accuracy:>9.3f} {score.macro_f1:>9.3f} "
            f"{score.majority_baseline:>9.3f} {score.lift_over_majority:>+8.3f}"
        )

    if evaluation.category_only_priority is not None:
        gain = evaluation.priority.accuracy - evaluation.category_only_priority
        lines += [
            "",
            f"Priority from the category alone: "
            f"{evaluation.category_only_priority:.3f}  "
            f"(model is {gain:+.3f} on top)",
            "  Topic is the easy half: VPN tickets skew high, printers skew low.",
            "  Everything above this line comes from reading how many people are",
            "  affected and how soon it matters -- not from recognising the topic.",
        ]

    if evaluation.by_language:
        lines += [
            "",
            "Per language:",
            f"{'lang':<6} {'n':>6} {'category':>9} {'priority':>9}",
        ]
        lines.append("-" * 33)
        for language, scores in evaluation.by_language.items():
            lines.append(
                f"{language:<6} {int(scores['size']):>6} "
                f"{scores['category_accuracy']:>9.3f} "
                f"{scores['priority_accuracy']:>9.3f}"
            )

    lines += [
        "",
        "Reminder: the test set is synthetic. These figures bound the pipeline,",
        "they do not predict accuracy on a real service desk.",
    ]
    return "\n".join(lines)
