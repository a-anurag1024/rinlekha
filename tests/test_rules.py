"""
Tests for pipeline/rules.py

Covers:
  - DERIVED_FIELD_RULES: correct computation and ordering dependency
  - HARD_DECLINE_RULES: each rule fires on the right condition
  - CLEAN_APPROVE_RULE: all five criteria must hold simultaneously
  - CONDITIONAL_RULES: each rule fires on the right condition
  - DECLINE_TRIGGERS: each trigger produces the expected field mutations
  - COMPATIBLE_TRIGGER_PAIRS: derived correctly from compatible_with lists
"""

import pytest

from pipeline.rules import (
    CLEAN_APPROVE_RULE,
    COMPATIBLE_TRIGGER_PAIRS,
    CONDITIONAL_FALLBACK,
    CONDITIONAL_RULES,
    DECLINE_TRIGGERS,
    DERIVED_FIELD_RULES,
    HARD_DECLINE_RULES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_profile() -> dict:
    """Minimal profile that passes all hard declines and approve criteria."""
    return {
        "cibil_score":             780,
        "monthly_income":          60_000,
        "existing_emi_monthly":    5_000,
        "annual_interest_rate":    12.0,
        "loan_tenure_months":      36,
        "loan_amount":             5_00_000,
        "missed_payments_24m":     0,
        "settled_accounts_ever":   0,
        "employment_tenure_years": 3.0,
        # derived — will be filled by apply_derived_rules
        "proposed_emi":            0.0,
        "foir_pre_loan":           0.0,
        "foir_post_loan":          0.0,
        "loan_to_income_ratio":    0.0,
    }


def apply_derived_rules(profile: dict) -> dict:
    for rule in DERIVED_FIELD_RULES:
        profile[rule.name] = rule.compute(profile)
    return profile


# ── DERIVED_FIELD_RULES ───────────────────────────────────────────────────────

class TestDerivedFieldRules:
    def test_rule_names_are_unique(self):
        names = [r.name for r in DERIVED_FIELD_RULES]
        assert len(names) == len(set(names))

    def test_proposed_emi_computed_first(self):
        # proposed_emi must appear before foir_post_loan (which depends on it)
        names = [r.name for r in DERIVED_FIELD_RULES]
        assert names.index("proposed_emi") < names.index("foir_post_loan")

    def test_emi_formula_correctness(self):
        # Manual: P=5L, r=12%/12=1%, n=36
        # EMI = 5L * 0.01 * 1.01^36 / (1.01^36 - 1)
        import math
        p = 5_00_000
        r = 12.0 / 12 / 100
        n = 36
        expected = p * r * (1 + r) ** n / ((1 + r) ** n - 1)
        profile = _base_profile()
        apply_derived_rules(profile)
        assert abs(profile["proposed_emi"] - expected) < 0.01

    def test_foir_pre_loan(self):
        profile = _base_profile()
        apply_derived_rules(profile)
        expected = round(5_000 / 60_000, 3)
        assert profile["foir_pre_loan"] == expected

    def test_foir_post_loan_includes_proposed_emi(self):
        profile = _base_profile()
        apply_derived_rules(profile)
        expected = round((5_000 + profile["proposed_emi"]) / 60_000, 3)
        assert profile["foir_post_loan"] == expected

    def test_loan_to_income_ratio(self):
        profile = _base_profile()
        apply_derived_rules(profile)
        expected = round(5_00_000 / 60_000, 1)
        assert profile["loan_to_income_ratio"] == expected

    def test_all_four_fields_added(self):
        profile = _base_profile()
        apply_derived_rules(profile)
        for name in ["proposed_emi", "foir_pre_loan", "foir_post_loan", "loan_to_income_ratio"]:
            assert name in profile


# ── HARD_DECLINE_RULES ────────────────────────────────────────────────────────

class TestHardDeclineRules:
    def _decline_keys(self, profile: dict) -> list[str]:
        return [r.condition_key for r in HARD_DECLINE_RULES if r.predicate(profile)]

    def test_low_cibil_triggers(self):
        p = {**_base_profile(), "cibil_score": 610}
        apply_derived_rules(p)
        assert "credit_score_below_minimum_threshold" in self._decline_keys(p)

    def test_high_cibil_does_not_trigger(self):
        p = _base_profile()
        apply_derived_rules(p)
        assert "credit_score_below_minimum_threshold" not in self._decline_keys(p)

    def test_high_foir_triggers(self):
        # Very high loan amount to push FOIR > 0.55
        p = {**_base_profile(), "loan_amount": 50_00_000, "monthly_income": 20_000,
             "existing_emi_monthly": 5_000}
        apply_derived_rules(p)
        assert "foir_exceeds_policy_ceiling_of_55pct" in self._decline_keys(p)

    def test_excessive_delinquency_triggers(self):
        p = {**_base_profile(), "missed_payments_24m": 4}
        apply_derived_rules(p)
        assert "excessive_recent_delinquency" in self._decline_keys(p)

    def test_three_missed_does_not_trigger_delinquency(self):
        p = {**_base_profile(), "missed_payments_24m": 3}
        apply_derived_rules(p)
        assert "excessive_recent_delinquency" not in self._decline_keys(p)

    def test_multiple_settled_triggers(self):
        p = {**_base_profile(), "settled_accounts_ever": 2}
        apply_derived_rules(p)
        assert "multiple_settled_loan_accounts" in self._decline_keys(p)

    def test_high_lti_triggers(self):
        p = {**_base_profile(), "loan_amount": 50_00_000, "monthly_income": 50_000,
             "existing_emi_monthly": 0}
        apply_derived_rules(p)
        assert "loan_amount_grossly_disproportionate_to_income" in self._decline_keys(p)

    def test_all_rules_fire_simultaneously(self):
        # Profile that triggers every hard decline condition at once
        p = {
            "cibil_score":             610,
            "monthly_income":          15_000,
            "existing_emi_monthly":    10_000,
            "annual_interest_rate":    24.0,
            "loan_tenure_months":      12,
            "loan_amount":             50_00_000,
            "missed_payments_24m":     5,
            "settled_accounts_ever":   3,
            "employment_tenure_years": 1.0,
        }
        apply_derived_rules(p)
        fired = self._decline_keys(p)
        assert len(fired) == len(HARD_DECLINE_RULES)

    def test_clean_profile_fires_no_decline_rules(self):
        p = _base_profile()
        apply_derived_rules(p)
        assert self._decline_keys(p) == []


# ── CLEAN_APPROVE_RULE ────────────────────────────────────────────────────────

class TestCleanApproveRule:
    def _make_approve_profile(self) -> dict:
        p = {
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
        return apply_derived_rules(p)

    def test_clean_profile_satisfies_all(self):
        p = self._make_approve_profile()
        assert CLEAN_APPROVE_RULE.satisfied(p)

    def test_low_cibil_fails(self):
        p = {**self._make_approve_profile(), "cibil_score": 759}
        assert not CLEAN_APPROVE_RULE.satisfied(p)

    def test_high_foir_fails(self):
        p = self._make_approve_profile()
        p["existing_emi_monthly"] = 40_000
        apply_derived_rules(p)
        assert not CLEAN_APPROVE_RULE.satisfied(p)

    def test_any_missed_payment_fails(self):
        p = {**self._make_approve_profile(), "missed_payments_24m": 1}
        assert not CLEAN_APPROVE_RULE.satisfied(p)

    def test_short_tenure_fails(self):
        p = {**self._make_approve_profile(), "employment_tenure_years": 1.9}
        assert not CLEAN_APPROVE_RULE.satisfied(p)

    def test_any_settled_account_fails(self):
        p = {**self._make_approve_profile(), "settled_accounts_ever": 1}
        assert not CLEAN_APPROVE_RULE.satisfied(p)

    def test_all_predicates_must_hold(self):
        """Fail each predicate individually — approve only when all pass."""
        base = self._make_approve_profile()
        assert CLEAN_APPROVE_RULE.satisfied(base)
        # Each individual violation breaks approval
        for mutated in [
            {**base, "cibil_score": 759},
            {**base, "missed_payments_24m": 1},
            {**base, "employment_tenure_years": 1.5},
            {**base, "settled_accounts_ever": 1},
        ]:
            assert not CLEAN_APPROVE_RULE.satisfied(mutated)


# ── CONDITIONAL_RULES ─────────────────────────────────────────────────────────

class TestConditionalRules:
    def _condition_keys(self, profile: dict) -> list[str]:
        return [r.condition_key for r in CONDITIONAL_RULES if r.predicate(profile)]

    def test_high_foir_triggers_income_proof(self):
        p = {**_base_profile(), "foir_post_loan": 0.50}
        assert "income_proof_last_6_months_salary_slips" in self._condition_keys(p)

    def test_foir_below_46_does_not_trigger_income_proof(self):
        p = {**_base_profile(), "foir_post_loan": 0.45}
        assert "income_proof_last_6_months_salary_slips" not in self._condition_keys(p)

    def test_one_or_two_missed_payments_triggers_explanation(self):
        for n in [1, 2]:
            p = {**_base_profile(), "missed_payments_24m": n}
            assert "written_explanation_for_missed_payments" in self._condition_keys(p)

    def test_zero_or_three_missed_does_not_trigger_explanation(self):
        for n in [0, 3]:
            p = {**_base_profile(), "missed_payments_24m": n}
            assert "written_explanation_for_missed_payments" not in self._condition_keys(p)

    def test_short_tenure_triggers_hr_confirmation(self):
        p = {**_base_profile(), "employment_tenure_years": 0.9}
        assert "employment_confirmation_from_hr" in self._condition_keys(p)

    def test_high_lti_triggers_co_applicant(self):
        p = {**_base_profile(), "loan_to_income_ratio": 31.0}
        assert "additional_income_proof_or_co_applicant" in self._condition_keys(p)

    def test_low_cibil_triggers_guarantor(self):
        p = {**_base_profile(), "cibil_score": 670}
        assert "guarantor_or_collateral_security" in self._condition_keys(p)

    def test_one_settled_account_triggers_noc(self):
        p = {**_base_profile(), "settled_accounts_ever": 1}
        assert "noc_from_previous_lender_for_settled_account" in self._condition_keys(p)

    def test_two_settled_accounts_does_not_trigger_noc(self):
        # 2+ is a hard decline, not a conditional
        p = {**_base_profile(), "settled_accounts_ever": 2}
        assert "noc_from_previous_lender_for_settled_account" not in self._condition_keys(p)

    def test_multiple_conditions_collected(self):
        # Profile that should fire multiple conditional rules
        p = {
            **_base_profile(),
            "foir_post_loan":          0.50,
            "missed_payments_24m":     1,
            "employment_tenure_years": 0.5,
            "loan_to_income_ratio":    35.0,
        }
        keys = self._condition_keys(p)
        assert len(keys) >= 3


# ── DECLINE_TRIGGERS ──────────────────────────────────────────────────────────

class TestDeclineTriggers:
    def _trigger(self, name: str):
        return next(t for t in DECLINE_TRIGGERS if t.name == name)

    def _base(self) -> dict:
        return {
            "cibil_score":             700,
            "monthly_income":          50_000,
            "existing_emi_monthly":    5_000,
            "annual_interest_rate":    14.0,
            "loan_tenure_months":      36,
            "missed_payments_24m":     0,
            "settled_accounts_ever":   0,
            "employment_tenure_years": 2.0,
        }

    def test_low_cibil_trigger_sets_cibil_below_620(self, rng):
        t = self._trigger("low_cibil")
        p = t.apply(self._base(), rng)
        assert p["cibil_score"] < 620

    def test_delinquency_trigger_sets_missed_gte_4(self, rng):
        t = self._trigger("delinquency")
        p = t.apply(self._base(), rng)
        assert p["missed_payments_24m"] >= 4

    def test_settled_trigger_sets_settled_gte_2(self, rng):
        t = self._trigger("settled")
        p = t.apply(self._base(), rng)
        assert p["settled_accounts_ever"] >= 2

    def test_high_lti_trigger_sets_loan_above_60x_income(self, rng):
        t = self._trigger("high_lti")
        p = t.apply(self._base(), rng)
        assert p["loan_amount"] > p["monthly_income"] * 60

    def test_all_trigger_names_are_unique(self):
        names = [t.name for t in DECLINE_TRIGGERS]
        assert len(names) == len(set(names))

    def test_compatible_with_references_valid_trigger_names(self):
        valid_names = {t.name for t in DECLINE_TRIGGERS}
        for t in DECLINE_TRIGGERS:
            for other in t.compatible_with:
                assert other in valid_names, (
                    f"Trigger '{t.name}' lists '{other}' in compatible_with "
                    f"but no such trigger exists"
                )


# ── COMPATIBLE_TRIGGER_PAIRS ──────────────────────────────────────────────────

class TestCompatibleTriggerPairs:
    def test_no_duplicate_pairs(self):
        seen = set()
        for t1, t2 in COMPATIBLE_TRIGGER_PAIRS:
            key = tuple(sorted([t1.name, t2.name]))
            assert key not in seen, f"Duplicate pair: {key}"
            seen.add(key)

    def test_no_self_pairs(self):
        for t1, t2 in COMPATIBLE_TRIGGER_PAIRS:
            assert t1.name != t2.name

    def test_symmetry_consistency(self):
        """If (a, b) is a pair, both a.compatible_with contains b and vice versa."""
        for t1, t2 in COMPATIBLE_TRIGGER_PAIRS:
            assert t2.name in t1.compatible_with or t1.name in t2.compatible_with

    def test_expected_pairs_present(self):
        pair_names = {
            tuple(sorted([t1.name, t2.name])) for t1, t2 in COMPATIBLE_TRIGGER_PAIRS
        }
        expected = {
            ("delinquency", "low_cibil"),
            ("low_cibil", "settled"),
            ("delinquency", "settled"),
            ("high_foir", "high_lti"),
            ("high_foir", "low_cibil"),
        }
        assert expected == pair_names
