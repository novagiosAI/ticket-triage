from .data import (
    Dataset,
    build_dataset,
    dataset_summary,
    deduplicate_against,
    train_test_split,
)
from .evaluate import (
    Evaluation,
    TaskScore,
    category_only_baseline,
    evaluate,
    format_evaluation,
    majority_baseline,
)
from .model import ARTIFACT_VERSION, TicketTriageModel, build_pipeline

__version__ = "0.1.0"

__all__ = [
    "TicketTriageModel",
    "build_pipeline",
    "ARTIFACT_VERSION",
    "Dataset",
    "build_dataset",
    "train_test_split",
    "deduplicate_against",
    "dataset_summary",
    "evaluate",
    "format_evaluation",
    "majority_baseline",
    "category_only_baseline",
    "Evaluation",
    "TaskScore",
]
