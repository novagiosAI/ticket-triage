# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-09-21

First public release.

### Added

- `TicketTriageModel` — two logistic-regression classifiers (category, priority) over
  combined character (`char_wb`, 2–5) and word TF-IDF features. Character n-grams are
  the deliberate choice: they survive the code-switching between Darija in Arabic
  script, Darija in Latin letters, French and English that word features fragment on.
- Trains in about 50 seconds on a laptop CPU; the artifact is **0.5 MB**, small
  enough for a free-tier deployment. No GPU, no external API.
- Training data generated entirely by
  [`glpi-testkit`](https://github.com/novagiosAI/glpi-testkit). **No customer ticket is
  involved at any point**, which is what makes the model publishable.
- Evaluation reporting three baselines beside the headline figure — majority class,
  category-only for priority, and per-language — because an accuracy number without
  them is not evidence.
- `train_test_split` deduplicates the test half against the training half by default,
  with `deduplicate=False` kept so the inflated numbers stay reproducible.
- `ARTIFACT_VERSION` check, so a stale artifact is refused rather than silently
  producing nonsense.
- `ticket-triage` CLI: `train`, `evaluate`, `predict` (with `--json`), `sample`.
- `app.py` — a Gradio demo that states its own limitations on the page.
- UTF-8 output forced on Windows consoles so Arabic does not raise
  `UnicodeEncodeError`.
- 39 tests. Passes `mypy --strict`.
- CI across Python 3.10–3.13 with `pytest`, `ruff` and `mypy`.

### Measured

On 9000 synthetic training tickets and 1430 deduplicated held-out ones:

| task | accuracy | macro-F1 | majority | topic only |
|---|---|---|---|---|
| category | 1.000 | 1.000 | 0.228 | — |
| priority | 0.901 | 0.885 | 0.406 | 0.513 |

Per language (category / priority): Arabic 1.000 / 0.954 · Darija 1.000 / 0.894 ·
French 1.000 / 0.887.

The category figure measures a task that is easy by construction — the categories use
disjoint vocabulary — and should be read as "the pipeline works", not as an accuracy
claim. The priority figure is the meaningful one, and it is reported against the 0.513
reachable from the topic alone.

### Fixed during development

- **Priority was unlearnable.** The generator sampled urgency and impact but wrote the
  body from the category alone, so the text carried no severity. A model trained on
  that output scored 0.469 against a 0.508 majority baseline — worse than guessing.
  Fixed upstream in `glpi-testkit` by phrasing severity the way users do; priority
  went from 0.469 to 0.901.
- **Train and test overlapped.** Template saturation — 15% unique texts at 20 000
  drawn — meant different random seeds still collided, and most of the test batch had
  been seen verbatim. Deduplication is now on by default.
- **`class_weight="balanced"` was the wrong default**, costing a third of the accuracy
  for macro-F1 that was not needed here. Now `None`, with the trade-off documented and
  selectable.
- **A metric was misnamed.** The category-only figure was labelled an information
  "ceiling"; once the generator expressed severity the model exceeded it, so the name
  was wrong. It is a baseline, and beating it is the point.

[Unreleased]: https://github.com/novagiosAI/ticket-triage/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/novagiosAI/ticket-triage/releases/tag/v0.1.0
