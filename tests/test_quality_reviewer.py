"""
Tests for pipeline/quality_reviewer.py

Covers:
  - extract_section: correct text extraction, missing header
  - extract_decision: all three outcomes, missing decision
  - quality_check: section presence, rec patterns, language,
                   risk-flag count, decision consistency,
                   structural_pass / needs_review aggregation
  - review_all: enrichment, length preservation
  - generate_report: counts, pass_rate, advisory, flagged_ids
  - save_report: JSON written, parent dirs created
"""

import json
import re
import textwrap

import pytest

from pipeline.quality_reviewer import (
    FORBIDDEN_PHRASES,
    REQUIRED_REC_PATTERNS,
    REQUIRED_SECTIONS,
    extract_decision,
    extract_section,
    generate_report,
    quality_check,
    review_all,
    save_report,
)


# ─── Memo fixtures ────────────────────────────────────────────────────────────

def _make_memo(
    *,
    sections: list[str] | None = None,
    decision: str = "APPROVE",
    risk_grade: str = "A",
    flag_count: int = 2,
    forbidden: bool = False,
) -> str:
    """
    Build a minimal but structurally valid memo.

    ``sections`` — if provided, only these section headers are included
    (default: all 6 REQUIRED_SECTIONS).
    """
    active_sections = sections if sections is not None else list(REQUIRED_SECTIONS)

    flags = "\n".join(f"- Risk flag {i}" for i in range(1, flag_count + 1))
    footer = "certainly" if forbidden else "assessment complete"

    lines: list[str] = []
    for header in active_sections:
        lines.append(header)
        if header == "## RISK FLAGS":
            lines.append(flags)
        elif header == "## RECOMMENDATION":
            lines.append(
                f"DECISION: {decision}\n"
                f"CONDITIONS: None\n"
                f"RISK GRADE: {risk_grade}\n"
                f"DECISION AUTHORITY: Branch Credit Manager\n"
                f"REVIEW TRIGGER: Material income change would alter assessment."
            )
        else:
            lines.append(f"Body text for {header[3:].lower()}. {footer}")
        lines.append("")  # blank line between sections

    return "\n".join(lines)


def _make_example(
    outcome: str = "APPROVE",
    decision: str = "APPROVE",
    **memo_kwargs,
) -> dict:
    """Wrap a memo in a minimal example dict."""
    return {
        "profile_id": f"p_{outcome.lower()}",
        "input_profile": {"outcome": outcome},
        "output_memo": _make_memo(decision=decision, **memo_kwargs),
    }


# ─── extract_section ──────────────────────────────────────────────────────────

class TestExtractSection:
    def test_returns_content_between_headers(self):
        memo = (
            "## RISK FLAGS\n"
            "- Flag A\n"
            "- Flag B\n"
            "\n"
            "## RECOMMENDATION\n"
            "DECISION: APPROVE\n"
        )
        section = extract_section(memo, "## RISK FLAGS")
        assert "Flag A" in section
        assert "Flag B" in section
        assert "DECISION" not in section

    def test_last_section_goes_to_eof(self):
        memo = "## ANALYST NOTES\nSome notes here.\n"
        section = extract_section(memo, "## ANALYST NOTES")
        assert "Some notes here." in section

    def test_missing_header_returns_empty_string(self):
        memo = "## APPLICANT SUMMARY\nContent.\n"
        assert extract_section(memo, "## DEBT SERVICEABILITY") == ""

    def test_does_not_bleed_into_next_section(self):
        memo = "## CREDIT BEHAVIOR\nCB text.\n\n## RISK FLAGS\nFlag.\n"
        section = extract_section(memo, "## CREDIT BEHAVIOR")
        assert "Flag." not in section


# ─── extract_decision ─────────────────────────────────────────────────────────

class TestExtractDecision:
    def test_approve(self):
        assert extract_decision("DECISION: APPROVE\n") == "APPROVE"

    def test_decline(self):
        assert extract_decision("DECISION: DECLINE\n") == "DECLINE"

    def test_conditional_approve_normalised(self):
        result = extract_decision("DECISION: CONDITIONAL APPROVE\n")
        assert result == "CONDITIONAL_APPROVE"

    def test_returns_none_when_missing(self):
        assert extract_decision("No decision here.") is None

    def test_ignores_case_in_surrounding_text(self):
        # Pattern is case-sensitive per spec; mixed case in body should not
        # produce a spurious match when value is valid
        assert extract_decision("decision: APPROVE") is None  # key must be uppercase

    def test_extracts_from_multiline_memo(self):
        memo = _make_memo(decision="DECLINE")
        assert extract_decision(memo) == "DECLINE"


# ─── quality_check ────────────────────────────────────────────────────────────

class TestQualityCheckSections:
    def test_all_sections_present_all_true(self):
        checks = quality_check(_make_example())
        for section in REQUIRED_SECTIONS:
            key = "section_" + section[3:].lower().replace(" ", "_")
            assert checks[key] is True, f"{key} should be True"

    def test_missing_section_flagged(self):
        sections_minus_one = [s for s in REQUIRED_SECTIONS if s != "## CREDIT BEHAVIOR"]
        ex = _make_example(sections=sections_minus_one)
        checks = quality_check(ex)
        assert checks["section_credit_behavior"] is False

    def test_structural_pass_false_when_section_missing(self):
        sections_minus_one = [s for s in REQUIRED_SECTIONS if s != "## RISK FLAGS"]
        ex = _make_example(sections=sections_minus_one)
        assert quality_check(ex)["structural_pass"] is False


class TestQualityCheckRecFields:
    def test_all_rec_patterns_present(self):
        checks = quality_check(_make_example())
        for suffix, _ in REQUIRED_REC_PATTERNS:
            assert checks[f"rec_{suffix}"] is True, f"rec_{suffix} should be True"

    def test_missing_risk_grade_flagged(self):
        # Memo with no RISK GRADE line
        memo = _make_memo().replace(
            f"RISK GRADE: A", "RISK LEVEL: A"
        )
        ex = {"profile_id": "p", "input_profile": {"outcome": "APPROVE"},
              "output_memo": memo}
        checks = quality_check(ex)
        assert checks["rec_risk_grade"] is False

    def test_invalid_risk_grade_flagged(self):
        memo = _make_memo(risk_grade="D")  # D is not in [ABC][+-]?
        ex = {"profile_id": "p", "input_profile": {"outcome": "APPROVE"},
              "output_memo": memo}
        checks = quality_check(ex)
        assert checks["rec_risk_grade"] is False

    @pytest.mark.parametrize("grade", ["A", "B+", "B", "B-", "C", "A+", "C-"])
    def test_valid_risk_grades_accepted(self, grade):
        memo = _make_memo(risk_grade=grade)
        ex = {"profile_id": "p", "input_profile": {"outcome": "APPROVE"},
              "output_memo": memo}
        assert quality_check(ex)["rec_risk_grade"] is True


class TestQualityCheckLanguage:
    def test_clean_memo_no_forbidden(self):
        checks = quality_check(_make_example())
        assert checks["no_forbidden_language"] is True
        assert checks["forbidden_phrases_found"] == []

    @pytest.mark.parametrize("phrase", FORBIDDEN_PHRASES)
    def test_each_forbidden_phrase_caught(self, phrase):
        memo = _make_memo() + f"\nExtra note: {phrase} this is certain."
        ex = {"profile_id": "p", "input_profile": {"outcome": "APPROVE"},
              "output_memo": memo}
        checks = quality_check(ex)
        assert checks["no_forbidden_language"] is False
        assert phrase in checks["forbidden_phrases_found"]

    def test_forbidden_language_does_not_affect_structural_pass(self):
        # Language hygiene is advisory — structural_pass should still be True
        ex = _make_example(forbidden=True)
        checks = quality_check(ex)
        assert checks["no_forbidden_language"] is False
        assert checks["structural_pass"] is True

    def test_forbidden_phrases_case_insensitive(self):
        memo = _make_memo() + "\nThis is DEFINITELY the case."
        ex = {"profile_id": "p", "input_profile": {"outcome": "APPROVE"},
              "output_memo": memo}
        checks = quality_check(ex)
        assert checks["no_forbidden_language"] is False


class TestQualityCheckRiskFlags:
    @pytest.mark.parametrize("count", [2, 3, 4])
    def test_valid_flag_count_passes(self, count):
        checks = quality_check(_make_example(flag_count=count))
        assert checks["risk_flags_count_valid"] is True
        assert checks["risk_flags_count"] == count

    @pytest.mark.parametrize("count", [0, 1, 5, 6])
    def test_invalid_flag_count_fails(self, count):
        checks = quality_check(_make_example(flag_count=count))
        assert checks["risk_flags_count_valid"] is False

    def test_flag_count_invalid_sets_structural_pass_false(self):
        checks = quality_check(_make_example(flag_count=1))
        assert checks["structural_pass"] is False


class TestQualityCheckDecisionConsistency:
    def test_matching_decision_approve(self):
        checks = quality_check(_make_example(outcome="APPROVE", decision="APPROVE"))
        assert checks["decision_matches_expected"] is True

    def test_matching_decision_decline(self):
        checks = quality_check(_make_example(outcome="DECLINE", decision="DECLINE"))
        assert checks["decision_matches_expected"] is True

    def test_matching_decision_conditional(self):
        checks = quality_check(
            _make_example(outcome="CONDITIONAL_APPROVE",
                          decision="CONDITIONAL APPROVE")
        )
        assert checks["decision_matches_expected"] is True

    def test_mismatched_decision_is_advisory_only(self):
        # Mismatch → decision_matches_expected=False but structural_pass unaffected
        checks = quality_check(_make_example(outcome="APPROVE", decision="DECLINE"))
        assert checks["decision_matches_expected"] is False
        # structural_pass is determined by critical checks only — still True here
        assert checks["structural_pass"] is True

    def test_no_decision_in_memo(self):
        memo = _make_memo().replace("DECISION: APPROVE\n", "")
        ex = {"profile_id": "p", "input_profile": {"outcome": "APPROVE"},
              "output_memo": memo}
        checks = quality_check(ex)
        # rec_decision fails → structural_pass also fails
        assert checks["rec_decision"] is False
        assert checks["structural_pass"] is False


class TestQualityCheckAggregation:
    def test_perfect_memo_passes(self):
        checks = quality_check(_make_example())
        assert checks["structural_pass"] is True
        assert checks["needs_review"] is False

    def test_needs_review_is_inverse_of_structural_pass(self):
        for flag_count in [1, 2, 3, 5]:
            checks = quality_check(_make_example(flag_count=flag_count))
            assert checks["needs_review"] is not checks["structural_pass"]

    def test_empty_memo_fails_all_structural(self):
        ex = {"profile_id": "p", "input_profile": {"outcome": "APPROVE"},
              "output_memo": ""}
        checks = quality_check(ex)
        assert checks["structural_pass"] is False
        assert checks["needs_review"] is True

    def test_none_memo_handled_gracefully(self):
        ex = {"profile_id": "p", "input_profile": {"outcome": "APPROVE"},
              "output_memo": None}
        checks = quality_check(ex)
        assert checks["structural_pass"] is False


# ─── review_all ───────────────────────────────────────────────────────────────

class TestReviewAll:
    def test_returns_same_count(self):
        examples = [_make_example() for _ in range(5)]
        results = review_all(examples)
        assert len(results) == 5

    def test_each_result_has_qc_key(self):
        examples = [_make_example() for _ in range(3)]
        for result in review_all(examples):
            assert "qc" in result

    def test_original_keys_preserved(self):
        ex = _make_example()
        result = review_all([ex])[0]
        assert "profile_id" in result
        assert "input_profile" in result
        assert "output_memo" in result

    def test_qc_contains_structural_pass(self):
        for result in review_all([_make_example()]):
            assert "structural_pass" in result["qc"]


# ─── generate_report ──────────────────────────────────────────────────────────

class TestGenerateReport:
    def _reviewed(self, n_pass: int, n_fail: int) -> list[dict]:
        results = []
        for i in range(n_pass):
            ex = _make_example()
            results.append({**ex, "qc": quality_check(ex),
                             "profile_id": f"pass_{i}"})
        for i in range(n_fail):
            # flag_count=1 → structural failure
            ex = _make_example(flag_count=1)
            results.append({**ex, "qc": quality_check(ex),
                             "profile_id": f"fail_{i}"})
        return results

    def test_total_count(self):
        report = generate_report(self._reviewed(7, 3))
        assert report["total"] == 10

    def test_passed_failed_sum(self):
        report = generate_report(self._reviewed(7, 3))
        assert report["passed"] == 7
        assert report["failed"] == 3

    def test_pass_rate(self):
        report = generate_report(self._reviewed(8, 2))
        assert abs(report["pass_rate"] - 0.8) < 0.001

    def test_flagged_ids_are_failures(self):
        report = generate_report(self._reviewed(5, 3))
        assert len(report["flagged_profile_ids"]) == 3
        assert all(pid.startswith("fail_") for pid in report["flagged_profile_ids"])

    def test_advisory_counts_present(self):
        report = generate_report(self._reviewed(3, 2))
        assert "advisory" in report
        assert "language_failures" in report["advisory"]
        assert "decision_mismatches" in report["advisory"]

    def test_empty_input_returns_zeros(self):
        report = generate_report([])
        assert report["total"] == 0
        assert report["pass_rate"] == 0.0

    def test_all_pass_no_flagged_ids(self):
        report = generate_report(self._reviewed(5, 0))
        assert report["failed"] == 0
        assert report["flagged_profile_ids"] == []


# ─── save_report ──────────────────────────────────────────────────────────────

class TestSaveReport:
    def test_writes_valid_json(self, tmp_path):
        report = {"total": 5, "passed": 4, "failed": 1, "pass_rate": 0.8}
        out = tmp_path / "qc_report.json"
        save_report(report, out)
        loaded = json.loads(out.read_text())
        assert loaded == report

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "report.json"
        save_report({"total": 0}, out)
        assert out.exists()

    def test_report_is_pretty_printed(self, tmp_path):
        out = tmp_path / "report.json"
        save_report({"total": 1}, out)
        content = out.read_text()
        # Pretty-printed JSON has newlines
        assert "\n" in content
