"""Command line entry point: ``ticket-triage``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .data import build_dataset, dataset_summary, train_test_split
from .evaluate import evaluate, format_evaluation
from .model import TicketTriageModel

DEFAULT_MODEL = "artifacts/model.joblib"


def _force_utf8() -> None:
    """Windows consoles default to a legacy code page and would raise on
    Arabic output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):  # pragma: no cover
                pass


def _cmd_train(args: argparse.Namespace) -> int:
    train, test = train_test_split(args.samples, seed=args.seed)
    print(f"train {len(train)} / test {len(test)} tickets (seed {args.seed})")

    started = time.time()
    model = TicketTriageModel.train(
        train.texts,
        train.categories,
        train.priorities,
        class_weight=args.class_weight,
        metadata={
            "samples": args.samples,
            "seed": args.seed,
            "class_weight": args.class_weight,
            "train_summary": dataset_summary(train),
            "data_source": "glpi-testkit synthetic — no client data",
        },
    )
    print(f"trained on CPU in {time.time() - started:.1f}s")

    evaluation = evaluate(model, test, train)
    print()
    print(format_evaluation(evaluation))

    model.metadata = {**(model.metadata or {}), "evaluation": evaluation.to_dict()}
    path = model.save(args.output)
    size_mb = path.stat().st_size / 1_000_000
    print()
    print(f"saved {path} ({size_mb:.1f} MB)")

    if args.report:
        Path(args.report).write_text(
            json.dumps(evaluation.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"wrote {args.report}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    model = TicketTriageModel.load(args.model)
    train, test = train_test_split(args.samples, seed=args.seed)
    print(format_evaluation(evaluate(model, test, train)))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    model = TicketTriageModel.load(args.model)
    text = args.text if args.text and args.text != "-" else sys.stdin.read()

    result = model.predict(text)
    if args.json:
        result["top_categories"] = model.top_categories(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"category : {result['category']}  ({result['category_confidence']:.0%})")
    print(f"priority : {result['priority']}  ({result['priority_confidence']:.0%})")
    print()
    print("runners-up:")
    for label, score in model.top_categories(text):
        print(f"  {label:<22} {score:.1%}")
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    dataset = build_dataset(args.count, seed=args.seed)
    for text, category, priority, language in zip(
        dataset.texts,
        dataset.categories,
        dataset.priorities,
        dataset.languages,
        strict=True,
    ):
        print(f"[{language}] {category} P{priority}")
        print(f"  {text.replace(chr(10), ' / ')}")
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ticket-triage",
        description=(
            "Multilingual ITSM ticket classifier trained on synthetic data only."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="generate data, train, evaluate, save")
    train.add_argument("-n", "--samples", type=int, default=8000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("-o", "--output", default=DEFAULT_MODEL)
    train.add_argument("--report", help="also write the metrics as JSON here")
    train.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        type=lambda value: None if value == "none" else value,
        help=(
            "none (default) maximises accuracy; balanced trades accuracy for "
            "recall on rare priorities"
        ),
    )
    train.set_defaults(func=_cmd_train)

    ev = sub.add_parser("evaluate", help="score a saved model")
    ev.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ev.add_argument("-n", "--samples", type=int, default=8000)
    ev.add_argument("--seed", type=int, default=42)
    ev.set_defaults(func=_cmd_evaluate)

    predict = sub.add_parser("predict", help="classify one ticket")
    predict.add_argument("text", nargs="?", help="ticket text, or '-' for stdin")
    predict.add_argument("-m", "--model", default=DEFAULT_MODEL)
    predict.add_argument("--json", action="store_true")
    predict.set_defaults(func=_cmd_predict)

    sample = sub.add_parser("sample", help="print generated tickets")
    sample.add_argument("-n", "--count", type=int, default=5)
    sample.add_argument("--seed", type=int, default=42)
    sample.set_defaults(func=_cmd_sample)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
