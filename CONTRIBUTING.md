# Contributing to ticket-triage

Thanks for taking the time to contribute.

## Getting set up

```bash
git clone https://github.com/novagiosAI/ticket-triage.git
cd ticket-triage
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Before opening a pull request

```bash
pytest
ruff check .
ruff format .
mypy
```

If you change anything that affects the model, **rerun `ticket-triage train` and
update the numbers in the README**. A README that disagrees with the code is worse
than one with no numbers in it.

## The rule that matters here

**Never train on real tickets in this repository, and never commit a model trained
on them.** The entire value of this project is that the published artifact provably
contains no customer data. One contaminated model destroys that, permanently and
invisibly.

## Reporting a number

An accuracy claim needs its baselines attached, or it is not a claim:

- **majority class** — if the model barely beats it, it learned nothing;
- **category only**, for priority — if the model barely beats it, it is recognising
  the topic and nothing else;
- **per language** — an aggregate hides a model that only works in French.

Two mistakes were made and fixed during the first release, and both are worth
knowing about:

1. Priority scored **below** the majority baseline, because the generator sampled
   urgency and impact but never expressed them in the text. The fix belonged in the
   data, not the model.
2. Train and test **overlapped**, because the template space saturates and different
   random seeds still collide. Deduplication is on by default now, and the inflated
   numbers stay reachable with `deduplicate=False` so the problem remains visible.

If a number looks too good, it usually is. Say so in the pull request.

## Changing the model

`model.py` is deliberately the only place that knows what the classifier is. The
CLI, evaluation and demo talk to `TicketTriageModel` and nothing else, so a
transformer implementation can replace it without touching them.

If you propose one, report the CPU inference time and the artifact size alongside
the accuracy. Both are constraints, not afterthoughts: the demo has to run on a
free CPU tier, and the model has to be installable on a server the client already
owns.

Bump `ARTIFACT_VERSION` whenever the feature pipeline changes shape, so an old
artifact is refused instead of silently producing nonsense.

## Adding to the data

The data comes from [`glpi-testkit`](https://github.com/novagiosAI/glpi-testkit) — new
categories and new severity phrasings belong there, not here.

## Security issues

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md). Note
in particular that loading a model artifact executes code.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
