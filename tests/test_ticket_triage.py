from __future__ import annotations

import json

import pytest

from ticket_triage import (
    ARTIFACT_VERSION,
    TicketTriageModel,
    build_dataset,
    category_only_baseline,
    dataset_summary,
    evaluate,
    format_evaluation,
    majority_baseline,
    train_test_split,
)

#: Small enough to keep the suite fast, large enough for the assertions below.
SAMPLES = 1200


@pytest.fixture(scope="module")
def split():
    return train_test_split(SAMPLES, seed=42)


@pytest.fixture(scope="module")
def trained(split):
    train, _ = split
    return TicketTriageModel.train(train.texts, train.categories, train.priorities)


class TestDataset:
    def test_is_reproducible(self):
        first = build_dataset(200, seed=7)
        second = build_dataset(200, seed=7)
        assert first.texts == second.texts
        assert first.priorities == second.priorities

    def test_different_seeds_differ(self):
        assert build_dataset(200, seed=1).texts != build_dataset(200, seed=2).texts

    def test_train_and_test_do_not_overlap(self, split):
        """Different seeds are not enough — the templates saturate, so the test
        half has to be deduplicated against the training half."""
        train, test = split
        assert not set(train.texts) & set(test.texts)

    def test_deduplication_actually_removes_rows(self):
        """If it ever stops removing anything, the overlap check has broken."""
        _, raw = train_test_split(SAMPLES, seed=42, deduplicate=False)
        _, clean = train_test_split(SAMPLES, seed=42, deduplicate=True)
        assert len(clean) < len(raw)

    def test_without_deduplication_the_halves_do_overlap(self):
        """Pins the problem itself, so the reason for dedup stays documented."""
        train, raw = train_test_split(SAMPLES, seed=42, deduplicate=False)
        assert set(train.texts) & set(raw.texts)

    def test_split_stays_within_the_requested_count(self, split):
        """Dedup only ever shrinks the test half; nothing is invented."""
        train, test = split
        assert 0 < len(test)
        assert len(train) + len(test) <= SAMPLES

    def test_all_three_languages_present(self, split):
        train, _ = split
        assert set(train.languages) == {"fr", "ar", "ary"}

    def test_text_joins_title_and_body(self):
        dataset = build_dataset(5, seed=1)
        assert all("\n" in text for text in dataset.texts)

    def test_subset_by_language(self, split):
        train, _ = split
        part = train.subset("ary")
        assert len(part) > 0
        assert set(part.languages) == {"ary"}

    def test_summary_counts_add_up(self, split):
        train, _ = split
        summary = dataset_summary(train)
        assert summary["size"] == len(train)
        assert sum(summary["languages"].values()) == len(train)

    @pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5])
    def test_invalid_test_fraction_is_refused(self, fraction):
        with pytest.raises(ValueError):
            train_test_split(100, test_fraction=fraction)

    def test_too_small_a_count_is_refused(self):
        with pytest.raises(ValueError):
            train_test_split(1, test_fraction=0.99)


class TestBaselines:
    def test_majority_baseline(self):
        assert majority_baseline(["a", "a", "a", "b"]) == 0.75

    def test_majority_baseline_of_nothing(self):
        assert majority_baseline([]) == 0.0

    def test_category_only_beats_majority(self, split):
        """Knowing the category has to help, or the label carries no signal."""
        train, test = split
        oracle = category_only_baseline(train, test)
        majority = majority_baseline(test.priorities)
        assert oracle > majority

    def test_category_only_is_below_one(self, split):
        """Topic alone cannot determine priority — if it ever does, the
        generator's urgency and impact ranges collapsed."""
        train, test = split
        assert category_only_baseline(train, test) < 1.0


class TestTraining:
    def test_refuses_empty_data(self):
        with pytest.raises(ValueError):
            TicketTriageModel.train([], [], [])

    def test_refuses_mismatched_lengths(self):
        with pytest.raises(ValueError):
            TicketTriageModel.train(["a", "b"], ["x"], [1, 2])

    def test_predicts_both_labels(self, trained):
        result = trained.predict("Impossible de se connecter au VPN")
        assert isinstance(result["category"], str)
        assert 1 <= result["priority"] <= 5
        assert 0.0 <= result["category_confidence"] <= 1.0
        assert 0.0 <= result["priority_confidence"] <= 1.0

    def test_predict_many(self, trained):
        results = trained.predict_many(["VPN en panne", "الطابعة معطلة"])
        assert len(results) == 2

    def test_top_categories_are_ranked(self, trained):
        ranked = trained.top_categories("bourrage papier imprimante", k=3)
        assert len(ranked) == 3
        assert ranked == sorted(ranked, key=lambda pair: -pair[1])

    def test_confidences_sum_to_one(self, trained):
        ranked = trained.top_categories("VPN", k=99)
        assert sum(score for _, score in ranked) == pytest.approx(1.0, abs=1e-6)

    def test_handles_all_three_languages(self, trained):
        for text in (
            "Impossible de se connecter au VPN",
            "تعذر الاتصال بالشبكة الافتراضية الخاصة",
            "l vpn kat9ta3 bzaf",
        ):
            assert trained.predict(text)["category"]

    def test_handles_empty_and_odd_input(self, trained):
        for text in ("", "   ", "???", "123456"):
            result = trained.predict(text)
            assert result["category"]


class TestEvaluation:
    def test_beats_the_majority_baseline_on_category(self, trained, split):
        _, test = split
        evaluation = evaluate(trained, test)
        assert evaluation.category.lift_over_majority > 0.3

    def test_priority_beats_the_category_only_baseline(self, trained, split):
        """The number that matters: the model must read severity from the text,
        not just recognise the topic."""
        train, test = split
        evaluation = evaluate(trained, test, train)
        assert evaluation.category_only_priority is not None
        assert evaluation.priority.accuracy > evaluation.category_only_priority

    def test_priority_beats_majority(self, trained, split):
        train, test = split
        evaluation = evaluate(trained, test, train)
        assert evaluation.priority.lift_over_majority > 0

    def test_no_language_is_left_behind(self, trained, split):
        """A single aggregate can hide a model that only works in French."""
        train, test = split
        evaluation = evaluate(trained, test, train)
        accuracies = [
            scores["category_accuracy"] for scores in evaluation.by_language.values()
        ]
        assert len(accuracies) == 3
        assert min(accuracies) > 0.8
        assert max(accuracies) - min(accuracies) < 0.15

    def test_category_only_baseline_is_omitted_without_a_train_set(
        self, trained, split
    ):
        _, test = split
        assert evaluate(trained, test).category_only_priority is None

    def test_empty_test_set_is_refused(self, trained):
        empty = build_dataset(1, seed=1)
        empty = type(empty)([], [], [], [])
        with pytest.raises(ValueError):
            evaluate(trained, empty)

    def test_report_is_json_serializable(self, trained, split):
        train, test = split
        payload = evaluate(trained, test, train).to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_formatted_report_names_its_limits(self, trained, split):
        """The summary must not be quotable out of context."""
        train, test = split
        text = format_evaluation(evaluate(trained, test, train))
        assert "synthetic" in text
        assert "category alone" in text


class TestPersistence:
    def test_round_trip(self, trained, tmp_path):
        path = trained.save(tmp_path / "model.joblib")
        assert path.exists()

        reloaded = TicketTriageModel.load(path)
        text = "Impossible de se connecter au VPN"
        assert reloaded.predict(text) == trained.predict(text)

    def test_creates_missing_directories(self, trained, tmp_path):
        path = trained.save(tmp_path / "nested" / "deeper" / "model.joblib")
        assert path.exists()

    def test_metadata_survives(self, split, tmp_path):
        train, _ = split
        model = TicketTriageModel.train(
            train.texts,
            train.categories,
            train.priorities,
            metadata={"note": "synthetic only"},
        )
        path = model.save(tmp_path / "m.joblib")
        assert TicketTriageModel.load(path).metadata["note"] == "synthetic only"

    def test_an_old_artifact_is_refused(self, trained, tmp_path):
        """Better a clear error than silent nonsense from a stale pipeline."""
        import joblib

        path = tmp_path / "old.joblib"
        joblib.dump(
            {
                "version": ARTIFACT_VERSION - 1,
                "category": trained.category,
                "priority": trained.priority,
                "metadata": {},
            },
            path,
        )
        with pytest.raises(ValueError, match="version"):
            TicketTriageModel.load(path)

    def test_artifact_stays_small_enough_to_ship(self, trained, tmp_path):
        """It has to fit in a free-tier deployment."""
        path = trained.save(tmp_path / "model.joblib")
        assert path.stat().st_size < 50_000_000
