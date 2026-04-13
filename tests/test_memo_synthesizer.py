"""
Tests for pipeline/memo_synthesizer.py

Covers:
  - _inr: Indian Rupee formatting
  - format_profile_as_readable_text: all key fields present and correctly formatted
  - _format_conditions: correct output per outcome class
  - build_synthesis_prompt: message structure, role, content completeness
  - synthesize_single_memo: success path, retry on transient errors,
    immediate failure on fatal errors, exhausted retries
  - synthesize_memo_batch: output structure (via Ray stub)
  - synthesize_all_memos: resumption skips completed ids, cost logging
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from pipeline.memo_synthesizer import (
    SYSTEM_PROMPT,
    _format_conditions,
    _get_decision_authority,
    _inr,
    build_synthesis_prompt,
    format_profile_as_readable_text,
    load_memos,
    save_memos,
    synthesize_all_memos,
    synthesize_memo_batch,
    synthesize_single_memo,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def approve_profile():
    return {
        "profile_id":             "profile_42_0001",
        "age":                    35,
        "city_tier":              "tier1",
        "employment_type":        "salaried_private",
        "sector":                 "IT",
        "monthly_income":         80_000,
        "employment_tenure_years": 4.5,
        "cibil_score":            790,
        "existing_emi_monthly":   5_000,
        "loan_accounts_active":   1,
        "missed_payments_24m":    0,
        "settled_accounts_ever":  0,
        "credit_vintage_years":   6.0,
        "loan_amount":            5_00_000,
        "loan_tenure_months":     36,
        "loan_purpose":           "home_renovation",
        "annual_interest_rate":   12.5,
        "proposed_emi":           16_734.0,
        "foir_pre_loan":          0.063,
        "foir_post_loan":         0.272,
        "loan_to_income_ratio":   6.3,
        "outcome":                "APPROVE",
        "conditions":             [],
    }


@pytest.fixture
def conditional_profile(approve_profile):
    return {
        **approve_profile,
        "profile_id":   "profile_42_0002",
        "cibil_score":  660,
        "missed_payments_24m": 1,
        "outcome":      "CONDITIONAL_APPROVE",
        "conditions":   [
            "written_explanation_for_missed_payments",
            "guarantor_or_collateral_security",
        ],
    }


@pytest.fixture
def decline_profile(approve_profile):
    return {
        **approve_profile,
        "profile_id":           "profile_42_0003",
        "cibil_score":          590,
        "missed_payments_24m":  5,
        "outcome":              "DECLINE",
        "conditions":           [
            "credit_score_below_minimum_threshold",
            "excessive_recent_delinquency",
        ],
    }


def _mock_openai_response(content: str, prompt_tokens: int = 580,
                           completion_tokens: int = 680) -> MagicMock:
    """Build a minimal OpenAI ChatCompletion response mock."""
    response           = MagicMock()
    response.choices[0].message.content = content
    response.usage.prompt_tokens         = prompt_tokens
    response.usage.completion_tokens     = completion_tokens
    return response


# ── _inr ──────────────────────────────────────────────────────────────────────

class TestInr:
    def test_small_amount(self):
        assert _inr(500) == "₹500"

    def test_thousands(self):
        result = _inr(5000)
        assert result.startswith("₹")
        assert "5" in result

    def test_lakhs(self):
        result = _inr(5_00_000)
        assert result.startswith("₹")
        assert "5" in result

    def test_always_starts_with_rupee(self):
        for amount in [0, 100, 10_000, 10_00_000]:
            assert _inr(amount).startswith("₹")


# ── format_profile_as_readable_text ──────────────────────────────────────────

class TestFormatProfileAsReadableText:
    def test_contains_cibil_score(self, approve_profile):
        text = format_profile_as_readable_text(approve_profile)
        assert "790" in text

    def test_contains_income(self, approve_profile):
        text = format_profile_as_readable_text(approve_profile)
        assert "80" in text  # ₹80,000

    def test_contains_foir_post_loan(self, approve_profile):
        text = format_profile_as_readable_text(approve_profile)
        assert "27.2" in text  # 0.272 → 27.2%

    def test_contains_foir_policy_ceiling(self, approve_profile):
        text = format_profile_as_readable_text(approve_profile)
        assert "55%" in text

    def test_contains_loan_amount(self, approve_profile):
        text = format_profile_as_readable_text(approve_profile)
        assert "5" in text  # ₹5,00,000

    def test_contains_employment_type(self, approve_profile):
        text = format_profile_as_readable_text(approve_profile)
        assert "Salaried Private" in text

    def test_contains_loan_purpose(self, approve_profile):
        text = format_profile_as_readable_text(approve_profile)
        assert "Home Renovation" in text

    def test_contains_all_section_headers(self, approve_profile):
        text = format_profile_as_readable_text(approve_profile)
        for header in ["DEMOGRAPHICS", "EMPLOYMENT", "CREDIT PROFILE",
                       "LOAN REQUEST", "DERIVED METRICS"]:
            assert header in text

    def test_missed_payments_present(self, approve_profile):
        text = format_profile_as_readable_text(approve_profile)
        assert "0" in text  # missed_payments_24m = 0

    def test_returns_string(self, approve_profile):
        assert isinstance(format_profile_as_readable_text(approve_profile), str)


# ── _format_conditions ────────────────────────────────────────────────────────

class TestFormatConditions:
    def test_approve_returns_none(self):
        assert _format_conditions("APPROVE", []) == "None"

    def test_empty_conditions_returns_none(self):
        assert _format_conditions("CONDITIONAL_APPROVE", []) == "None"

    def test_conditional_produces_numbered_list(self):
        result = _format_conditions("CONDITIONAL_APPROVE", [
            "written_explanation_for_missed_payments",
            "guarantor_or_collateral_security",
        ])
        assert "1." in result
        assert "2." in result

    def test_known_key_renders_as_readable_label(self):
        result = _format_conditions("CONDITIONAL_APPROVE", [
            "written_explanation_for_missed_payments"
        ])
        assert "missed payments" in result.lower()

    def test_unknown_key_falls_back_to_underscores_replaced(self):
        result = _format_conditions("DECLINE", ["some_unknown_condition"])
        assert "some unknown condition" in result.lower()

    def test_decline_reasons_rendered(self):
        result = _format_conditions("DECLINE", [
            "credit_score_below_minimum_threshold",
            "excessive_recent_delinquency",
        ])
        assert "CIBIL" in result or "620" in result
        assert "delinquency" in result.lower()


# ── _get_decision_authority ───────────────────────────────────────────────────

class TestGetDecisionAuthority:
    def _profile(self, outcome, loan_amount):
        return {"outcome": outcome, "loan_amount": loan_amount}

    # DECLINE always → HO Credit Committee regardless of amount
    def test_decline_small_loan_is_ho(self):
        assert _get_decision_authority(self._profile("DECLINE", 50_000)) == "HO Credit Committee"

    def test_decline_large_loan_is_ho(self):
        assert _get_decision_authority(self._profile("DECLINE", 5_000_000)) == "HO Credit Committee"

    # APPROVE + large loan → HO Credit Committee
    def test_approve_over_25L_is_ho(self):
        assert _get_decision_authority(self._profile("APPROVE", 2_600_000)) == "HO Credit Committee"

    # APPROVE + medium loan → Regional Credit Head
    def test_approve_over_10L_is_regional(self):
        assert _get_decision_authority(self._profile("APPROVE", 1_100_000)) == "Regional Credit Head"

    # CONDITIONAL_APPROVE → Regional Credit Head (regardless of size if ≤ ₹25L)
    def test_conditional_small_loan_is_regional(self):
        assert _get_decision_authority(self._profile("CONDITIONAL_APPROVE", 80_000)) == "Regional Credit Head"

    def test_conditional_medium_loan_is_regional(self):
        assert _get_decision_authority(self._profile("CONDITIONAL_APPROVE", 1_500_000)) == "Regional Credit Head"

    # APPROVE + small loan → Branch Credit Manager
    def test_approve_small_loan_is_branch(self):
        assert _get_decision_authority(self._profile("APPROVE", 5_00_000)) == "Branch Credit Manager"

    def test_approve_exactly_10L_is_branch(self):
        # Boundary: ≤ ₹10L is Branch
        assert _get_decision_authority(self._profile("APPROVE", 1_000_000)) == "Branch Credit Manager"

    def test_approve_exactly_25L_is_regional(self):
        # Boundary: exactly ₹25L is Regional (> 10L)
        assert _get_decision_authority(self._profile("APPROVE", 2_500_000)) == "Regional Credit Head"


# ── build_synthesis_prompt ────────────────────────────────────────────────────

class TestBuildSynthesisPrompt:
    def test_returns_two_messages(self, approve_profile):
        msgs = build_synthesis_prompt(approve_profile)
        assert len(msgs) == 2

    def test_first_message_is_system(self, approve_profile):
        msgs = build_synthesis_prompt(approve_profile)
        assert msgs[0]["role"] == "system"

    def test_second_message_is_user(self, approve_profile):
        msgs = build_synthesis_prompt(approve_profile)
        assert msgs[1]["role"] == "user"

    def test_system_content_matches_constant(self, approve_profile):
        msgs = build_synthesis_prompt(approve_profile)
        assert msgs[0]["content"] == SYSTEM_PROMPT

    def test_user_content_contains_section_headers(self, approve_profile):
        msgs = build_synthesis_prompt(approve_profile)
        user = msgs[1]["content"]
        for section in ["## APPLICANT SUMMARY", "## DEBT SERVICEABILITY",
                        "## CREDIT BEHAVIOR", "## RISK FLAGS",
                        "## RECOMMENDATION", "## ANALYST NOTES"]:
            assert section in user

    def test_user_content_contains_outcome(self, approve_profile):
        msgs = build_synthesis_prompt(approve_profile)
        assert "APPROVE" in msgs[1]["content"]

    def test_user_content_contains_cibil(self, approve_profile):
        msgs = build_synthesis_prompt(approve_profile)
        assert "790" in msgs[1]["content"]

    def test_conditional_profile_includes_conditions(self, conditional_profile):
        msgs = build_synthesis_prompt(conditional_profile)
        user = msgs[1]["content"]
        assert "missed payments" in user.lower()

    def test_decline_profile_includes_decline_reasons(self, decline_profile):
        msgs = build_synthesis_prompt(decline_profile)
        user = msgs[1]["content"]
        assert "DECLINE" in user

    def test_user_content_contains_decision_authority(self, approve_profile):
        msgs = build_synthesis_prompt(approve_profile)
        user = msgs[1]["content"]
        authority = _get_decision_authority(approve_profile)
        assert authority in user

    def test_decision_authority_varies_by_profile(
        self, approve_profile, decline_profile
    ):
        approve_auth = build_synthesis_prompt(approve_profile)[1]["content"]
        decline_auth = build_synthesis_prompt(decline_profile)[1]["content"]
        # A small-loan APPROVE vs any DECLINE must produce different authorities
        assert "Branch Credit Manager" in approve_auth or "Regional Credit Head" in approve_auth
        assert "HO Credit Committee" in decline_auth

    def test_system_prompt_is_identical_across_profiles(
        self, approve_profile, conditional_profile, decline_profile
    ):
        """System prompt must be identical so OpenAI caches it."""
        msgs = [
            build_synthesis_prompt(approve_profile)[0]["content"],
            build_synthesis_prompt(conditional_profile)[0]["content"],
            build_synthesis_prompt(decline_profile)[0]["content"],
        ]
        assert len(set(msgs)) == 1


# ── synthesize_single_memo ────────────────────────────────────────────────────

class TestSynthesizeSingleMemo:
    def test_success_path(self, approve_profile):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(
            "## APPLICANT SUMMARY\nTest memo content."
        )
        result = synthesize_single_memo(approve_profile, client)

        assert result["synthesis_status"] == "success"
        assert result["profile_id"] == "profile_42_0001"
        assert result["output_memo"] == "## APPLICANT SUMMARY\nTest memo content."
        assert result["input_tokens"] == 580
        assert result["output_tokens"] == 680
        assert result["input_profile"] is approve_profile

    def test_returns_all_required_keys(self, approve_profile):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("memo")
        result = synthesize_single_memo(approve_profile, client)
        for key in ["profile_id", "input_profile", "output_memo",
                    "input_tokens", "output_tokens", "synthesis_status"]:
            assert key in result

    def test_retries_on_rate_limit_error(self, approve_profile):
        from openai import RateLimitError
        client = MagicMock()
        # Fail twice, succeed on third attempt
        client.chat.completions.create.side_effect = [
            RateLimitError("rate limit", response=MagicMock(status_code=429), body={}),
            RateLimitError("rate limit", response=MagicMock(status_code=429), body={}),
            _mock_openai_response("memo after retry"),
        ]
        with patch("pipeline.memo_synthesizer.time.sleep"):
            result = synthesize_single_memo(
                approve_profile, client, max_retries=3, base_delay=0.0
            )
        assert result["synthesis_status"] == "success"
        assert client.chat.completions.create.call_count == 3

    def test_returns_error_after_max_retries_exhausted(self, approve_profile):
        from openai import RateLimitError
        client = MagicMock()
        client.chat.completions.create.side_effect = RateLimitError(
            "rate limit", response=MagicMock(status_code=429), body={}
        )
        with patch("pipeline.memo_synthesizer.time.sleep"):
            result = synthesize_single_memo(
                approve_profile, client, max_retries=2, base_delay=0.0
            )
        assert result["synthesis_status"].startswith("error:")
        assert result["output_memo"] is None
        assert client.chat.completions.create.call_count == 3  # initial + 2 retries

    def test_no_retry_on_auth_error(self, approve_profile):
        from openai import AuthenticationError
        client = MagicMock()
        client.chat.completions.create.side_effect = AuthenticationError(
            "bad key", response=MagicMock(status_code=401), body={}
        )
        result = synthesize_single_memo(approve_profile, client, max_retries=3)
        assert result["synthesis_status"].startswith("error:")
        assert client.chat.completions.create.call_count == 1  # no retry

    def test_no_retry_on_bad_request(self, approve_profile):
        from openai import BadRequestError
        client = MagicMock()
        client.chat.completions.create.side_effect = BadRequestError(
            "bad input", response=MagicMock(status_code=400), body={}
        )
        result = synthesize_single_memo(approve_profile, max_retries=3, client=client)
        assert result["synthesis_status"].startswith("error:")
        assert client.chat.completions.create.call_count == 1

    def test_retries_on_server_error(self, approve_profile):
        from openai import APIStatusError
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            APIStatusError("server error",
                           response=MagicMock(status_code=500), body={}),
            _mock_openai_response("recovered memo"),
        ]
        with patch("pipeline.memo_synthesizer.time.sleep"):
            result = synthesize_single_memo(
                approve_profile, client, max_retries=2, base_delay=0.0
            )
        assert result["synthesis_status"] == "success"

    def test_error_result_has_zero_tokens(self, approve_profile):
        from openai import RateLimitError
        client = MagicMock()
        client.chat.completions.create.side_effect = RateLimitError(
            "limit", response=MagicMock(status_code=429), body={}
        )
        with patch("pipeline.memo_synthesizer.time.sleep"):
            result = synthesize_single_memo(
                approve_profile, client, max_retries=0
            )
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0


# ── synthesize_memo_batch ─────────────────────────────────────────────────────

class TestSynthesizeMemosBatch:
    def test_returns_one_result_per_profile(
        self, approve_profile, conditional_profile
    ):
        with patch("pipeline.memo_synthesizer.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = (
                _mock_openai_response("memo")
            )
            # Ray stub runs synchronously
            results = synthesize_memo_batch(
                [approve_profile, conditional_profile],
                api_key="test-key",
            )
        assert len(results) == 2

    def test_profile_ids_preserved(self, approve_profile, conditional_profile):
        with patch("pipeline.memo_synthesizer.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = (
                _mock_openai_response("memo")
            )
            results = synthesize_memo_batch(
                [approve_profile, conditional_profile],
                api_key="test-key",
            )
        ids = {r["profile_id"] for r in results}
        assert ids == {"profile_42_0001", "profile_42_0002"}


# ── synthesize_all_memos ──────────────────────────────────────────────────────

class TestSynthesizeAllMemos:
    def _profiles(self, n: int = 4) -> list[dict]:
        base = {
            "age": 30, "city_tier": "tier1", "employment_type": "salaried_private",
            "sector": "IT", "monthly_income": 60000, "employment_tenure_years": 3.0,
            "cibil_score": 750, "existing_emi_monthly": 5000, "loan_accounts_active": 1,
            "missed_payments_24m": 0, "settled_accounts_ever": 0,
            "credit_vintage_years": 4.0, "loan_amount": 300000,
            "loan_tenure_months": 36, "loan_purpose": "education",
            "annual_interest_rate": 13.0, "proposed_emi": 10000,
            "foir_pre_loan": 0.083, "foir_post_loan": 0.25,
            "loan_to_income_ratio": 5.0, "outcome": "APPROVE", "conditions": [],
        }
        return [{**base, "profile_id": f"profile_0_{i:04d}"} for i in range(n)]

    def test_returns_results_for_all_profiles(self):
        profiles = self._profiles(4)
        with patch("pipeline.memo_synthesizer.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = (
                _mock_openai_response("memo")
            )
            results = synthesize_all_memos(
                profiles, api_key="test-key", n_workers=2
            )
        assert len(results) == 4

    def test_resumption_skips_completed_ids(self):
        profiles = self._profiles(4)
        completed = {"profile_0_0000", "profile_0_0001"}
        with patch("pipeline.memo_synthesizer.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = (
                _mock_openai_response("memo")
            )
            results = synthesize_all_memos(
                profiles, api_key="test-key", n_workers=2,
                completed_ids=completed,
            )
        # Only 2 new results — the 2 already-completed ones are skipped
        assert len(results) == 2
        result_ids = {r["profile_id"] for r in results}
        assert result_ids.isdisjoint(completed)

    def test_empty_pending_returns_empty_list(self):
        profiles = self._profiles(2)
        completed = {p["profile_id"] for p in profiles}
        results = synthesize_all_memos(
            profiles, api_key="test-key", completed_ids=completed
        )
        assert results == []


# ── save_memos / load_memos ───────────────────────────────────────────────────

class TestSaveLoadMemos:
    def test_round_trip(self, tmp_path, approve_profile):
        memos = [
            {"profile_id": "p1", "input_profile": approve_profile,
             "output_memo": "memo text", "synthesis_status": "success",
             "input_tokens": 100, "output_tokens": 200},
        ]
        path = tmp_path / "memos.jsonl"
        save_memos(memos, path)
        loaded = load_memos(path)
        assert len(loaded) == 1
        assert loaded[0]["profile_id"] == "p1"
        assert loaded[0]["output_memo"] == "memo text"

    def test_creates_parent_directories(self, tmp_path):
        memos = [{"profile_id": "x", "output_memo": "m",
                  "synthesis_status": "success",
                  "input_tokens": 0, "output_tokens": 0, "input_profile": {}}]
        path = tmp_path / "subdir" / "nested" / "memos.jsonl"
        save_memos(memos, path)
        assert path.exists()

    def test_each_line_is_valid_json(self, tmp_path, approve_profile):
        memos = [
            {"profile_id": f"p{i}", "output_memo": f"memo {i}",
             "synthesis_status": "success", "input_tokens": 0,
             "output_tokens": 0, "input_profile": approve_profile}
            for i in range(5)
        ]
        path = tmp_path / "memos.jsonl"
        save_memos(memos, path)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            json.loads(line)  # must not raise
