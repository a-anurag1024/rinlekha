"""
Tests for pipeline/samplers.py

Covers:
  - Each sampler class produces values within declared bounds
  - PROFILE_SCHEMA contains all expected fields
  - sample_base_profile returns a complete profile dict
  - Overrides replace schema samplers correctly
  - Realistic distribution samplers are wired to the correct fields
"""

import random

import pytest

from pipeline.samplers import (
    Choice,
    ClampedNormalSampler,
    Fixed,
    LogNormalSampler,
    PROFILE_SCHEMA,
    UniformFloat,
    UniformInt,
    WeibullSampler,
    ZeroInflatedSampler,
    sample_base_profile,
)

N_SAMPLES = 500  # enough for distribution checks without being slow


# ── UniformInt ────────────────────────────────────────────────────────────────

class TestUniformInt:
    def test_within_bounds(self, rng):
        s = UniformInt(10, 20)
        values = [s.sample(rng) for _ in range(N_SAMPLES)]
        assert all(10 <= v <= 20 for v in values)

    def test_returns_int(self, rng):
        s = UniformInt(0, 5)
        assert isinstance(s.sample(rng), int)

    def test_full_range_covered(self, rng):
        s = UniformInt(0, 5)
        values = {s.sample(rng) for _ in range(N_SAMPLES)}
        assert values == {0, 1, 2, 3, 4, 5}


# ── UniformFloat ──────────────────────────────────────────────────────────────

class TestUniformFloat:
    def test_within_bounds(self, rng):
        s = UniformFloat(1.0, 10.0, precision=2)
        values = [s.sample(rng) for _ in range(N_SAMPLES)]
        assert all(1.0 <= v <= 10.0 for v in values)

    def test_precision_respected(self, rng):
        s = UniformFloat(0.0, 1.0, precision=3)
        values = [s.sample(rng) for _ in range(N_SAMPLES)]
        # No value should have more than 3 decimal places
        assert all(round(v, 3) == v for v in values)


# ── Choice ────────────────────────────────────────────────────────────────────

class TestChoice:
    def test_only_returns_listed_options(self, rng):
        options = ["a", "b", "c"]
        s = Choice(options)
        values = {s.sample(rng) for _ in range(N_SAMPLES)}
        assert values.issubset(set(options))

    def test_all_options_reachable(self, rng):
        options = ["x", "y", "z"]
        s = Choice(options)
        values = {s.sample(rng) for _ in range(N_SAMPLES)}
        assert values == set(options)


# ── Fixed ─────────────────────────────────────────────────────────────────────

class TestFixed:
    def test_always_returns_same_value(self, rng):
        s = Fixed(42)
        assert all(s.sample(rng) == 42 for _ in range(20))

    def test_works_with_zero(self, rng):
        s = Fixed(0)
        assert s.sample(rng) == 0


# ── LogNormalSampler ──────────────────────────────────────────────────────────

class TestLogNormalSampler:
    def test_respects_lower_bound(self, rng):
        s = LogNormalSampler(mu=10.0, sigma=1.0, lo=5_000)
        values = [s.sample(rng) for _ in range(N_SAMPLES)]
        assert all(v >= 5_000 for v in values)

    def test_respects_upper_bound(self, rng):
        s = LogNormalSampler(mu=10.0, sigma=1.0, hi=1_00_000)
        values = [s.sample(rng) for _ in range(N_SAMPLES)]
        assert all(v <= 1_00_000 for v in values)

    def test_median_near_exp_mu(self, rng):
        import math
        mu = 10.2
        s = LogNormalSampler(mu=mu, sigma=0.7, lo=1)
        values = sorted(s.sample(rng) for _ in range(N_SAMPLES))
        median = values[N_SAMPLES // 2]
        expected_median = math.exp(mu)
        # Median should be within 20% of exp(mu)
        assert abs(median - expected_median) / expected_median < 0.20

    def test_precision_respected(self, rng):
        s = LogNormalSampler(mu=10.0, sigma=0.5, precision=0)
        values = [s.sample(rng) for _ in range(100)]
        assert all(v == round(v, 0) for v in values)


# ── ClampedNormalSampler ──────────────────────────────────────────────────────

class TestClampedNormalSampler:
    def test_always_within_bounds(self, rng):
        s = ClampedNormalSampler(mu=700, sigma=60, lo=550, hi=900)
        values = [s.sample(rng) for _ in range(N_SAMPLES)]
        assert all(550 <= v <= 900 for v in values)

    def test_mean_near_mu(self, rng):
        s = ClampedNormalSampler(mu=700, sigma=60, lo=550, hi=900)
        values = [s.sample(rng) for _ in range(N_SAMPLES)]
        mean = sum(values) / len(values)
        assert abs(mean - 700) < 20  # within 20 points


# ── ZeroInflatedSampler ───────────────────────────────────────────────────────

class TestZeroInflatedSampler:
    def test_zero_fraction_approximately_correct(self, rng):
        zero_prob = 0.70
        s = ZeroInflatedSampler(zero_prob=zero_prob, base=UniformInt(1, 6))
        values = [s.sample(rng) for _ in range(N_SAMPLES)]
        actual_zero_frac = sum(1 for v in values if v == 0) / N_SAMPLES
        # Allow ±10pp tolerance
        assert abs(actual_zero_frac - zero_prob) < 0.10

    def test_non_zero_values_come_from_base(self, rng):
        s = ZeroInflatedSampler(zero_prob=0.5, base=UniformInt(10, 20))
        values = [v for v in (s.sample(rng) for _ in range(N_SAMPLES)) if v != 0]
        assert all(10 <= v <= 20 for v in values)


# ── WeibullSampler ────────────────────────────────────────────────────────────

class TestWeibullSampler:
    def test_respects_lower_bound(self, rng):
        s = WeibullSampler(shape=0.9, scale=5.0, lo=0.25)
        values = [s.sample(rng) for _ in range(N_SAMPLES)]
        assert all(v >= 0.25 for v in values)

    def test_respects_upper_bound(self, rng):
        s = WeibullSampler(shape=0.9, scale=5.0, lo=0.0, hi=25.0)
        values = [s.sample(rng) for _ in range(N_SAMPLES)]
        assert all(v <= 25.0 for v in values)

    def test_shape_lt_1_skewed_early(self, rng):
        # shape < 1 → most mass at low values; median should be well below scale
        s = WeibullSampler(shape=0.9, scale=5.0, lo=0.0)
        values = sorted(s.sample(rng) for _ in range(N_SAMPLES))
        median = values[N_SAMPLES // 2]
        assert median < 5.0  # median below scale for shape < 1


# ── PROFILE_SCHEMA ────────────────────────────────────────────────────────────

EXPECTED_FIELDS = {
    "age", "city_tier", "employment_type", "monthly_income",
    "employment_tenure_years", "sector", "cibil_score",
    "existing_emi_monthly", "loan_accounts_active", "missed_payments_24m",
    "settled_accounts_ever", "credit_vintage_years", "loan_amount",
    "loan_tenure_months", "loan_purpose", "annual_interest_rate",
}

class TestProfileSchema:
    def test_all_expected_fields_present(self):
        assert EXPECTED_FIELDS.issubset(set(PROFILE_SCHEMA.keys()))

    def test_realistic_samplers_wired_to_correct_fields(self):
        assert isinstance(PROFILE_SCHEMA["monthly_income"],          LogNormalSampler)
        assert isinstance(PROFILE_SCHEMA["loan_amount"],             LogNormalSampler)
        assert isinstance(PROFILE_SCHEMA["employment_tenure_years"], WeibullSampler)
        assert isinstance(PROFILE_SCHEMA["credit_vintage_years"],    WeibullSampler)


# ── sample_base_profile ───────────────────────────────────────────────────────

class TestSampleBaseProfile:
    def test_returns_all_schema_fields(self, rng):
        p = sample_base_profile(rng)
        assert set(p.keys()) == set(PROFILE_SCHEMA.keys())

    def test_override_replaces_schema_sampler(self, rng):
        p = sample_base_profile(rng, overrides={"cibil_score": Fixed(800)})
        assert p["cibil_score"] == 800

    def test_override_does_not_affect_other_fields(self, rng):
        p = sample_base_profile(rng, overrides={"cibil_score": Fixed(800)})
        # All other fields should still be present
        other_fields = set(PROFILE_SCHEMA.keys()) - {"cibil_score"}
        assert other_fields.issubset(set(p.keys()))

    def test_multiple_overrides(self, rng):
        p = sample_base_profile(rng, overrides={
            "cibil_score":         Fixed(750),
            "missed_payments_24m": Fixed(0),
            "settled_accounts_ever": Fixed(0),
        })
        assert p["cibil_score"] == 750
        assert p["missed_payments_24m"] == 0
        assert p["settled_accounts_ever"] == 0

    def test_deterministic_with_same_seed(self):
        p1 = sample_base_profile(random.Random(42))
        p2 = sample_base_profile(random.Random(42))
        assert p1 == p2

    def test_different_seeds_produce_different_profiles(self):
        p1 = sample_base_profile(random.Random(1))
        p2 = sample_base_profile(random.Random(2))
        assert p1 != p2
