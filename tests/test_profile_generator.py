"""
Tests for pipeline/profile_generator.py

Covers:
  - compute_derived_fields: consistency with raw inputs
  - determine_outcome: DECLINE / APPROVE / CONDITIONAL routing
  - _sample_approve: all profiles receive APPROVE decision
  - _sample_decline: all profiles receive DECLINE; multi-condition coverage
  - _sample_conditional: all profiles receive CONDITIONAL_APPROVE
  - generate_profile_batch: output count, ID format, distribution
"""

import random

import pytest

from pipeline.profile_generator import (
    _sample_approve,
    _sample_conditional,
    _sample_decline,
    compute_derived_fields,
    determine_outcome,
    generate_profile_batch,
)

N_PROFILES = 200  # per outcome class — enough to catch systematic failures


# ── compute_derived_fields ────────────────────────────────────────────────────

class TestComputeDerivedFields:
    def _raw(self, **overrides) -> dict:
        base = {
            "monthly_income":       60_000,
            "existing_emi_monthly": 5_000,
            "annual_interest_rate": 12.0,
            "loan_tenure_months":   36,
            "loan_amount":          5_00_000,
        }
        return {**base, **overrides}

    def test_proposed_emi_positive(self):
        p = compute_derived_fields(self._raw())
        assert p["proposed_emi"] > 0

    def test_foir_pre_equals_existing_over_income(self):
        p = compute_derived_fields(self._raw())
        expected = round(5_000 / 60_000, 3)
        assert p["foir_pre_loan"] == expected

    def test_foir_post_greater_than_pre(self):
        p = compute_derived_fields(self._raw())
        assert p["foir_post_loan"] > p["foir_pre_loan"]

    def test_lti_equals_loan_over_income(self):
        p = compute_derived_fields(self._raw())
        assert p["loan_to_income_ratio"] == round(5_00_000 / 60_000, 1)

    def test_derived_fields_added_to_profile(self):
        p = compute_derived_fields(self._raw())
        for field in ["proposed_emi", "foir_pre_loan", "foir_post_loan", "loan_to_income_ratio"]:
            assert field in p

    def test_emi_consistent_with_foir(self):
        p = compute_derived_fields(self._raw())
        recomputed_foir = round(
            (p["existing_emi_monthly"] + p["proposed_emi"]) / p["monthly_income"], 3
        )
        assert p["foir_post_loan"] == recomputed_foir


# ── determine_outcome ─────────────────────────────────────────────────────────

class TestDetermineOutcome:
    def _profile(self, **overrides) -> dict:
        base = {
            "cibil_score":             780,
            "monthly_income":          1_00_000,
            "existing_emi_monthly":    5_000,
            "annual_interest_rate":    12.0,
            "loan_tenure_months":      36,
            "loan_amount":             5_00_000,
            "missed_payments_24m":     0,
            "settled_accounts_ever":   0,
            "employment_tenure_years": 3.0,
        }
        p = {**base, **overrides}
        return compute_derived_fields(p)

    # DECLINE cases
    def test_low_cibil_is_decline(self):
        outcome, _ = determine_outcome(self._profile(cibil_score=610))
        assert outcome == "DECLINE"

    def test_high_foir_is_decline(self):
        p = self._profile(monthly_income=20_000, existing_emi_monthly=8_000,
                          loan_amount=50_00_000)
        outcome, _ = determine_outcome(p)
        assert outcome == "DECLINE"

    def test_excessive_delinquency_is_decline(self):
        outcome, _ = determine_outcome(self._profile(missed_payments_24m=4))
        assert outcome == "DECLINE"

    def test_multiple_settled_is_decline(self):
        outcome, _ = determine_outcome(self._profile(settled_accounts_ever=2))
        assert outcome == "DECLINE"

    def test_high_lti_is_decline(self):
        p = self._profile(monthly_income=15_000, loan_amount=50_00_000,
                          existing_emi_monthly=0)
        outcome, _ = determine_outcome(p)
        assert outcome == "DECLINE"

    def test_multiple_decline_conditions_all_reported(self):
        p = self._profile(cibil_score=610, missed_payments_24m=5)
        outcome, conditions = determine_outcome(p)
        assert outcome == "DECLINE"
        assert len(conditions) >= 2
        assert "credit_score_below_minimum_threshold" in conditions
        assert "excessive_recent_delinquency" in conditions

    # APPROVE case
    def test_clean_profile_is_approve(self):
        outcome, conditions = determine_outcome(self._profile())
        assert outcome == "APPROVE"
        assert conditions == []

    # CONDITIONAL cases
    def test_borderline_foir_is_conditional(self):
        # Target FOIR ~0.54: just below the 0.55 hard-decline ceiling but
        # above the 0.38 clean-approve threshold.
        # With income=40k, existing_emi=5k, loan=5L @ 12%/36m:
        #   EMI ≈ 16,607 → FOIR = (5k + 16.6k) / 40k ≈ 0.54
        p = self._profile(monthly_income=40_000, existing_emi_monthly=5_000,
                          loan_amount=5_00_000)
        outcome, _ = determine_outcome(p)
        assert outcome == "CONDITIONAL_APPROVE"

    def test_one_missed_payment_is_conditional(self):
        p = self._profile(missed_payments_24m=1)
        outcome, conditions = determine_outcome(p)
        assert outcome == "CONDITIONAL_APPROVE"
        assert "written_explanation_for_missed_payments" in conditions

    def test_conditional_returns_at_least_one_condition(self):
        p = self._profile(missed_payments_24m=1)
        _, conditions = determine_outcome(p)
        assert len(conditions) >= 1

    def test_fallback_condition_when_nothing_triggered(self):
        # Profile just below APPROVE threshold (CIBIL 760 exactly, but let's use 759)
        p = self._profile(cibil_score=759)
        outcome, conditions = determine_outcome(p)
        assert outcome == "CONDITIONAL_APPROVE"
        # Should still get at least the fallback
        assert len(conditions) >= 1


# ── _sample_approve ───────────────────────────────────────────────────────────

class TestSampleApprove:
    def test_always_returns_approve(self, rng):
        for _ in range(N_PROFILES):
            p = _sample_approve(rng)
            assert p is not None
            assert p["outcome"] == "APPROVE"

    def test_cibil_gte_760(self, rng):
        profiles = [_sample_approve(rng) for _ in range(N_PROFILES)]
        assert all(p["cibil_score"] >= 760 for p in profiles if p)

    def test_foir_post_lte_038(self, rng):
        profiles = [_sample_approve(rng) for _ in range(N_PROFILES)]
        assert all(p["foir_post_loan"] <= 0.38 for p in profiles if p)

    def test_zero_missed_payments(self, rng):
        profiles = [_sample_approve(rng) for _ in range(N_PROFILES)]
        assert all(p["missed_payments_24m"] == 0 for p in profiles if p)

    def test_zero_settled_accounts(self, rng):
        profiles = [_sample_approve(rng) for _ in range(N_PROFILES)]
        assert all(p["settled_accounts_ever"] == 0 for p in profiles if p)

    def test_tenure_gte_2_years(self, rng):
        profiles = [_sample_approve(rng) for _ in range(N_PROFILES)]
        assert all(p["employment_tenure_years"] >= 2.0 for p in profiles if p)

    def test_conditions_list_is_empty(self, rng):
        profiles = [_sample_approve(rng) for _ in range(N_PROFILES)]
        assert all(p["conditions"] == [] for p in profiles if p)

    def test_all_required_fields_present(self, rng):
        p = _sample_approve(rng)
        required = {
            "cibil_score", "monthly_income", "loan_amount", "foir_post_loan",
            "outcome", "conditions", "proposed_emi",
        }
        assert required.issubset(p.keys())


# ── _sample_decline ───────────────────────────────────────────────────────────

class TestSampleDecline:
    def _generate(self, rng, n=N_PROFILES):
        profiles = []
        for _ in range(n):
            p = _sample_decline(rng)
            p = compute_derived_fields(p)
            outcome, conditions = determine_outcome(p)
            p["outcome"] = outcome
            p["conditions"] = conditions
            profiles.append(p)
        return profiles

    def test_vast_majority_are_decline(self, rng):
        profiles = self._generate(rng)
        n_decline = sum(1 for p in profiles if p["outcome"] == "DECLINE")
        # Allow a small fraction of edge-case near-misses
        assert n_decline / len(profiles) >= 0.90

    def test_multi_condition_declines_present(self, rng):
        profiles = self._generate(rng)
        declines = [p for p in profiles if p["outcome"] == "DECLINE"]
        multi = [p for p in declines if len(p["conditions"]) > 1]
        assert len(multi) > 0, "Expected some multi-condition decline profiles"

    def test_single_condition_declines_present(self, rng):
        profiles = self._generate(rng)
        declines = [p for p in profiles if p["outcome"] == "DECLINE"]
        single = [p for p in declines if len(p["conditions"]) == 1]
        assert len(single) > 0

    def test_all_hard_decline_reasons_appear(self, rng):
        from pipeline.rules import HARD_DECLINE_RULES
        profiles = self._generate(rng, n=500)
        declines = [p for p in profiles if p["outcome"] == "DECLINE"]
        all_conditions = {c for p in declines for c in p["conditions"]}
        expected = {r.condition_key for r in HARD_DECLINE_RULES}
        assert expected == all_conditions, (
            f"Missing decline reasons: {expected - all_conditions}"
        )


# ── _sample_conditional ───────────────────────────────────────────────────────

class TestSampleConditional:
    def test_always_returns_conditional_approve(self, rng):
        for _ in range(N_PROFILES):
            p = _sample_conditional(rng)
            assert p is not None
            assert p["outcome"] == "CONDITIONAL_APPROVE"

    def test_conditions_list_non_empty(self, rng):
        profiles = [_sample_conditional(rng) for _ in range(N_PROFILES)]
        assert all(len(p["conditions"]) >= 1 for p in profiles if p)

    def test_cibil_in_conditional_range(self, rng):
        profiles = [_sample_conditional(rng) for _ in range(N_PROFILES)]
        # CIBIL override is 620-759; derived fields may push to CONDITIONAL anyway
        assert all(p["cibil_score"] < 760 for p in profiles if p)

    def test_all_required_fields_present(self, rng):
        p = _sample_conditional(rng)
        required = {
            "cibil_score", "monthly_income", "loan_amount", "foir_post_loan",
            "outcome", "conditions", "proposed_emi",
        }
        assert required.issubset(p.keys())


# ── generate_profile_batch ────────────────────────────────────────────────────

class TestGenerateProfileBatch:
    def test_returns_correct_total_count(self):
        # Ray is stubbed — remote functions run synchronously
        profiles = generate_profile_batch(5, 10, 5, seed=42)
        assert len(profiles) == 20  # 5+10+5; some may be None-filtered

    def test_profile_ids_contain_seed(self):
        profiles = generate_profile_batch(3, 3, 3, seed=1234)
        assert all("1234" in p["profile_id"] for p in profiles)

    def test_profile_ids_unique(self):
        profiles = generate_profile_batch(5, 10, 5, seed=99)
        ids = [p["profile_id"] for p in profiles]
        assert len(ids) == len(set(ids))

    def test_outcome_distribution_matches_requested(self):
        from collections import Counter
        profiles = generate_profile_batch(20, 55, 25, seed=7)
        dist = Counter(p["outcome"] for p in profiles)
        # Allow ±5 for edge-case reclassification
        assert abs(dist.get("APPROVE", 0) - 20) <= 5
        assert abs(dist.get("CONDITIONAL_APPROVE", 0) - 55) <= 5
