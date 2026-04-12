"""
Tests for pipeline/dataset_builder.py

Covers:
  - get_cibil_band: all four bands and boundary values
  - get_foir_band: all four bands and boundary values
  - format_as_instruction_pair: required keys, content correctness,
                                metadata fields populated
  - filter_qc_passed: qc-enriched examples, raw examples, mixed
  - stratified_split: correct total count, per-label proportion,
                      determinism, all splits present, single-class input
  - save_splits / load_split: JSONL round-trip, file naming, parent dirs
  - build_dataset: integration (filter → format → split → save)
"""

import json
import random

import pytest

from pipeline.dataset_builder import (
    INSTRUCTION,
    SPLIT_RATIOS,
    build_dataset,
    filter_qc_passed,
    format_as_instruction_pair,
    get_cibil_band,
    get_foir_band,
    load_split,
    save_splits,
    stratified_split,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_profile(
    outcome: str = "APPROVE",
    cibil: int = 780,
    foir: float = 0.30,
) -> dict:
    return {
        "profile_id": f"p_{outcome.lower()}_{cibil}",
        "age": 32,
        "city_tier": "tier1",
        "employment_type": "salaried",
        "sector": "private",
        "employment_tenure_years": 3.0,
        "monthly_income": 60_000,
        "existing_emi_monthly": 5_000,
        "annual_interest_rate": 12.0,
        "loan_tenure_months": 36,
        "loan_amount": 3_00_000,
        "loan_purpose": "home_improvement",
        "cibil_score": cibil,
        "missed_payments_24m": 0,
        "settled_accounts_ever": 0,
        "loan_accounts_active": 1,
        "credit_vintage_years": 4.0,
        "proposed_emi": 9_963,
        "foir_pre_loan": round(5_000 / 60_000, 3),
        "foir_post_loan": foir,
        "loan_to_income_ratio": round(3_00_000 / 60_000, 1),
        "outcome": outcome,
        "conditions": [],
    }


def _make_example(outcome: str = "APPROVE", qc_pass: bool | None = None) -> dict:
    profile = _make_profile(outcome=outcome)
    ex: dict = {
        "profile_id": profile["profile_id"],
        "input_profile": profile,
        "output_memo": f"## APPLICANT SUMMARY\nSample memo for {outcome}.",
        "synthesis_status": "success",
    }
    if qc_pass is not None:
        ex["qc"] = {"structural_pass": qc_pass, "needs_review": not qc_pass}
    return ex


def _examples(n_approve=4, n_conditional=6, n_decline=2) -> list[dict]:
    result = []
    for _ in range(n_approve):
        result.append(_make_example("APPROVE"))
    for _ in range(n_conditional):
        result.append(_make_example("CONDITIONAL_APPROVE"))
    for _ in range(n_decline):
        result.append(_make_example("DECLINE"))
    return result


# ─── get_cibil_band ───────────────────────────────────────────────────────────

class TestGetCibilBand:
    @pytest.mark.parametrize("score,expected", [
        (900, "excellent"),
        (760, "excellent"),
        (759, "good"),
        (700, "good"),
        (699, "fair"),
        (620, "fair"),
        (619, "poor"),
        (300, "poor"),
    ])
    def test_band_boundaries(self, score, expected):
        assert get_cibil_band(score) == expected


# ─── get_foir_band ────────────────────────────────────────────────────────────

class TestGetFoirBand:
    @pytest.mark.parametrize("foir,expected", [
        (0.00, "low"),
        (0.30, "low"),
        (0.31, "moderate"),
        (0.45, "moderate"),
        (0.46, "high"),
        (0.55, "high"),
        (0.56, "excessive"),
        (0.90, "excessive"),
    ])
    def test_band_boundaries(self, foir, expected):
        assert get_foir_band(foir) == expected


# ─── format_as_instruction_pair ───────────────────────────────────────────────

class TestFormatAsInstructionPair:
    def test_required_keys_present(self):
        pair = format_as_instruction_pair(_make_example())
        required = {
            "instruction", "input", "output",
            "outcome", "cibil_band", "foir_band",
            "employment_type", "loan_purpose", "profile_id",
        }
        assert required.issubset(pair.keys())

    def test_instruction_is_constant(self):
        assert format_as_instruction_pair(_make_example())["instruction"] == INSTRUCTION

    def test_output_is_memo_text(self):
        ex = _make_example()
        pair = format_as_instruction_pair(ex)
        assert pair["output"] == ex["output_memo"]

    def test_outcome_copied_from_profile(self):
        for outcome in ("APPROVE", "CONDITIONAL_APPROVE", "DECLINE"):
            ex = _make_example(outcome)
            assert format_as_instruction_pair(ex)["outcome"] == outcome

    def test_cibil_band_correct(self):
        ex = _make_example()
        ex["input_profile"]["cibil_score"] = 750
        pair = format_as_instruction_pair(ex)
        assert pair["cibil_band"] == "good"

    def test_foir_band_correct(self):
        ex = _make_example()
        ex["input_profile"]["foir_post_loan"] = 0.35
        pair = format_as_instruction_pair(ex)
        assert pair["foir_band"] == "moderate"

    def test_employment_type_copied(self):
        ex = _make_example()
        assert format_as_instruction_pair(ex)["employment_type"] == "salaried"

    def test_loan_purpose_copied(self):
        ex = _make_example()
        assert format_as_instruction_pair(ex)["loan_purpose"] == "home_improvement"

    def test_profile_id_copied(self):
        ex = _make_example("APPROVE")
        pair = format_as_instruction_pair(ex)
        assert pair["profile_id"] == ex["input_profile"]["profile_id"]

    def test_input_contains_profile_text(self):
        ex = _make_example()
        pair = format_as_instruction_pair(ex)
        # format_profile_as_readable_text produces section headers
        assert "LOAN REQUEST" in pair["input"] or "DEMOGRAPHICS" in pair["input"]


# ─── filter_qc_passed ─────────────────────────────────────────────────────────

class TestFilterQcPassed:
    def test_keeps_qc_pass(self):
        examples = [_make_example(qc_pass=True) for _ in range(5)]
        assert len(filter_qc_passed(examples)) == 5

    def test_removes_qc_fail(self):
        examples = [_make_example(qc_pass=True)] * 3 + [_make_example(qc_pass=False)] * 2
        assert len(filter_qc_passed(examples)) == 3

    def test_keeps_raw_examples_without_qc(self):
        # Examples with no qc key should pass through (pre-QC pipeline)
        examples = [_make_example() for _ in range(4)]
        assert len(filter_qc_passed(examples)) == 4

    def test_mixed_batch(self):
        examples = (
            [_make_example(qc_pass=True)] * 2
            + [_make_example(qc_pass=False)] * 2
            + [_make_example()] * 2   # no qc key — kept
        )
        assert len(filter_qc_passed(examples)) == 4


# ─── stratified_split ─────────────────────────────────────────────────────────

class TestStratifiedSplit:
    def test_total_count_preserved(self):
        examples = _examples(10, 20, 10)
        splits = stratified_split(
            [format_as_instruction_pair(e) for e in examples],
            label_key="outcome",
        )
        total = sum(len(v) for v in splits.values())
        assert total == 40

    def test_all_split_names_present(self):
        splits = stratified_split(
            [format_as_instruction_pair(e) for e in _examples(10, 10, 10)],
        )
        assert set(splits.keys()) == set(SPLIT_RATIOS.keys())

    def test_no_examples_lost(self):
        formatted = [format_as_instruction_pair(e) for e in _examples(8, 16, 8)]
        splits = stratified_split(formatted)
        ids_in = {e["profile_id"] for e in formatted}
        ids_out = {e["profile_id"] for s in splits.values() for e in s}
        assert ids_in == ids_out

    def test_each_class_appears_in_train(self):
        formatted = [format_as_instruction_pair(e) for e in _examples(10, 20, 10)]
        splits = stratified_split(formatted)
        outcomes_in_train = {e["outcome"] for e in splits["train"]}
        assert "APPROVE" in outcomes_in_train
        assert "CONDITIONAL_APPROVE" in outcomes_in_train
        assert "DECLINE" in outcomes_in_train

    def test_train_is_largest_split(self):
        formatted = [format_as_instruction_pair(e) for e in _examples(10, 20, 10)]
        splits = stratified_split(formatted)
        assert len(splits["train"]) > len(splits["validation"])
        assert len(splits["train"]) > len(splits["test"])

    def test_deterministic_with_same_seed(self):
        formatted = [format_as_instruction_pair(e) for e in _examples(10, 20, 10)]
        s1 = stratified_split(formatted, seed=7)
        s2 = stratified_split(formatted, seed=7)
        assert [e["profile_id"] for e in s1["train"]] == [e["profile_id"] for e in s2["train"]]

    def test_different_seeds_produce_different_order(self):
        formatted = [format_as_instruction_pair(e) for e in _examples(20, 40, 20)]
        s1 = stratified_split(formatted, seed=1)
        s2 = stratified_split(formatted, seed=99)
        # Highly unlikely to be identical with 80 training examples
        assert [e["profile_id"] for e in s1["train"]] != [e["profile_id"] for e in s2["train"]]

    def test_single_class_still_splits(self):
        examples = [format_as_instruction_pair(_make_example("APPROVE")) for _ in range(20)]
        # Give each a unique profile_id
        for i, ex in enumerate(examples):
            ex["profile_id"] = f"p_{i}"
        splits = stratified_split(examples)
        assert sum(len(v) for v in splits.values()) == 20

    def test_approximate_80_10_10_ratio(self):
        formatted = [format_as_instruction_pair(e) for e in _examples(40, 80, 40)]
        splits = stratified_split(formatted)
        total = len(formatted)
        # Allow ±5% absolute deviation
        assert abs(len(splits["train"]) / total - 0.80) <= 0.05
        assert abs(len(splits["validation"]) / total - 0.10) <= 0.05
        assert abs(len(splits["test"]) / total - 0.10) <= 0.05


# ─── save_splits / load_split ─────────────────────────────────────────────────

class TestSaveSplitsLoadSplit:
    def _splits(self) -> dict[str, list[dict]]:
        formatted = [format_as_instruction_pair(e) for e in _examples(4, 8, 4)]
        return stratified_split(formatted)

    def test_creates_one_file_per_split(self, tmp_path):
        splits = self._splits()
        save_splits(splits, tmp_path)
        for name in splits:
            assert (tmp_path / f"{name}.jsonl").exists()

    def test_round_trip(self, tmp_path):
        splits = self._splits()
        save_splits(splits, tmp_path)
        for name, rows in splits.items():
            loaded = load_split(tmp_path / f"{name}.jsonl")
            assert len(loaded) == len(rows)
            assert loaded[0].keys() == rows[0].keys()

    def test_each_line_is_valid_json(self, tmp_path):
        splits = self._splits()
        save_splits(splits, tmp_path)
        for name in splits:
            path = tmp_path / f"{name}.jsonl"
            for line in path.read_text().splitlines():
                json.loads(line)  # must not raise

    def test_creates_parent_dirs(self, tmp_path):
        splits = self._splits()
        out = tmp_path / "nested" / "data"
        save_splits(splits, out)
        assert (out / "train.jsonl").exists()

    def test_profile_id_preserved_in_loaded(self, tmp_path):
        splits = self._splits()
        save_splits(splits, tmp_path)
        for name, rows in splits.items():
            loaded = load_split(tmp_path / f"{name}.jsonl")
            original_ids = {r["profile_id"] for r in rows}
            loaded_ids = {r["profile_id"] for r in loaded}
            assert original_ids == loaded_ids


# ─── build_dataset ────────────────────────────────────────────────────────────

class TestBuildDataset:
    def test_returns_expected_split_keys(self, tmp_path):
        examples = _examples(10, 20, 10)
        splits = build_dataset(examples, output_dir=tmp_path)
        assert set(splits.keys()) == {"train", "validation", "test"}

    def test_output_files_created(self, tmp_path):
        build_dataset(_examples(10, 20, 10), output_dir=tmp_path)
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "validation.jsonl").exists()
        assert (tmp_path / "test.jsonl").exists()

    def test_qc_failures_excluded(self, tmp_path):
        examples = (
            [_make_example(qc_pass=True)] * 30
            + [_make_example(qc_pass=False)] * 10
        )
        splits = build_dataset(examples, output_dir=tmp_path)
        total = sum(len(v) for v in splits.values())
        assert total == 30

    def test_all_formatted_rows_have_instruction(self, tmp_path):
        splits = build_dataset(_examples(5, 10, 5), output_dir=tmp_path)
        for rows in splits.values():
            assert all(r["instruction"] == INSTRUCTION for r in rows)

    def test_no_examples_lost_without_qc(self, tmp_path):
        examples = _examples(10, 20, 10)
        splits = build_dataset(examples, output_dir=tmp_path)
        total = sum(len(v) for v in splits.values())
        assert total == 40
