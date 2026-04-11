"""
pipeline/samplers.py — Field sampler primitives and the canonical profile schema.

DESIGN
------
Each field in the borrower profile has a FieldSampler that owns the sampling
logic for that field. The schema dict maps field names to sampler instances.
Swapping a distribution (e.g. income from uniform → log-normal) is a one-line
change here — no other file needs to change.

profile_generator.py imports `PROFILE_SCHEMA` and `sample_base_profile`. The
targeted outcome-class samplers (_sample_approve etc.) pass `overrides` to
`sample_base_profile` to constrain specific fields without re-implementing
the rest.

CURRENT STATE
─────────────
Four fields use fitted Indian macro-distributions (v1):
  monthly_income        — LogNormal(mu=10.2, σ=0.7)   median ₹26.7k
  loan_amount           — LogNormal(mu=11.9, σ=0.9)   median ₹1.48L
  employment_tenure_yrs — Weibull(k=0.9, λ=5.0)       median 3.2 yrs
  credit_vintage_years  — Weibull(k=1.2, λ=4.0)       median 2.9 yrs

Remaining fields are uniform — either schema-ineffective (always overridden
by outcome-class samplers), categorical, or acceptable approximations.
See PROFILE_SCHEMA inline comments for the full breakdown.

REMAINING EXTENSIONS (schema-ineffective until sampler architecture is updated)
─────────────────────────────────────────────────────────────────────────────────
These fields are always overridden in profile_generator.py outcome-class
samplers, so changing their schema distribution has no effect today.
Realising them requires introducing a ContextualFieldSampler that receives
the partially-built profile as context (see NOTE ON DEPENDENT FIELD SAMPLING
in rules.py).

  cibil_score           — ClampedNormal(mu=700, σ=60, lo=550, hi=900)
                          (TransUnion CIBIL Industry Insights 2023)

  existing_emi_monthly  — ZeroInflated(zero_prob=0.35,
                            base=LogNormal(mu=9.5, σ=0.7, lo=500))
                          ~35% of applicants have no existing EMI.
                          Currently set as income fraction — needs
                          ContextualFieldSampler to use schema.

  missed_payments_24m   — ZeroInflated(zero_prob=0.70,
                            base=Poisson(λ=1.2, lo=1, hi=6))
                          ~70% of applicants have 0 missed payments.

  settled_accounts_ever — ZeroInflated(zero_prob=0.85, base=UniformInt(1,3))
                          ~85% of applicants have 0 settled accounts.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ─── FieldSampler protocol ────────────────────────────────────────────────────

@runtime_checkable
class FieldSampler(Protocol):
    """Any object with a .sample(rng) method qualifies as a FieldSampler."""
    def sample(self, rng: random.Random) -> Any: ...


# ─── Concrete sampler classes ─────────────────────────────────────────────────

@dataclass
class UniformInt:
    """Uniform integer in [lo, hi] inclusive."""
    lo: int
    hi: int

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.lo, self.hi)


@dataclass
class UniformFloat:
    """Uniform float in [lo, hi], rounded to `precision` decimal places."""
    lo: float
    hi: float
    precision: int = 2

    def sample(self, rng: random.Random) -> float:
        return round(rng.uniform(self.lo, self.hi), self.precision)


@dataclass
class Choice:
    """Uniform random choice from a fixed list of options."""
    options: list[Any]

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.options)


@dataclass
class Fixed:
    """Always returns the same value. Used in overrides to pin a field."""
    value: Any

    def sample(self, rng: random.Random) -> Any:
        return self.value


# ─── Realistic distribution sampler classes ──────────────────────────────────
# All classes below are fully implemented and wired into PROFILE_SCHEMA for the
# fields where they have effect. See the schema section for which fields still
# use uniform distributions and why.

@dataclass
class LogNormalSampler:
    """
    Log-normal distribution: X = exp(Normal(mu, sigma)).
    `mu` and `sigma` are in log-space.

    To fit from real data:
        import numpy as np
        mu = np.mean(np.log(observations))
        sigma = np.std(np.log(observations))
    """
    mu: float
    sigma: float
    lo: float | None = None
    hi: float | None = None
    precision: int = 0

    def sample(self, rng: random.Random) -> float:
        val = math.exp(rng.gauss(self.mu, self.sigma))
        if self.lo is not None:
            val = max(val, self.lo)
        if self.hi is not None:
            val = min(val, self.hi)
        return round(val, self.precision)


@dataclass
class ClampedNormalSampler:
    """Normal distribution clamped to [lo, hi]. Uses rejection sampling."""
    mu: float
    sigma: float
    lo: float
    hi: float
    precision: int = 0
    _max_retries: int = field(default=50, repr=False)

    def sample(self, rng: random.Random) -> float:
        for _ in range(self._max_retries):
            val = rng.gauss(self.mu, self.sigma)
            if self.lo <= val <= self.hi:
                return round(val, self.precision)
        # Fallback: clamp (happens rarely at distribution tails)
        return round(max(self.lo, min(self.hi, rng.gauss(self.mu, self.sigma))), self.precision)


@dataclass
class ZeroInflatedSampler:
    """
    Zero-inflated distribution: returns 0 with probability `zero_prob`,
    otherwise delegates to `base` sampler.
    Useful for existing_emi_monthly, missed_payments, settled_accounts.
    """
    zero_prob: float
    base: FieldSampler

    def sample(self, rng: random.Random) -> Any:
        if rng.random() < self.zero_prob:
            return 0
        return self.base.sample(rng)


@dataclass
class WeibullSampler:
    """
    Weibull distribution for right-skewed positive quantities
    (employment tenure, credit vintage).
    shape=k < 1 → heavy early peak (high churn); k > 1 → peaked interior.
    """
    shape: float   # k in the standard Weibull parameterisation
    scale: float   # λ (characteristic life)
    lo: float = 0.0
    hi: float | None = None
    precision: int = 2

    def sample(self, rng: random.Random) -> float:
        # Python's weibullvariate uses shape=alpha, scale=beta
        val = rng.weibullvariate(self.scale, self.shape)
        val = max(val, self.lo)
        if self.hi is not None:
            val = min(val, self.hi)
        return round(val, self.precision)


# ─── Canonical profile schema ─────────────────────────────────────────────────
#
# SCHEMA EFFECTIVENESS NOTE
# ─────────────────────────
# Not every field here actually influences generated profiles — the targeted
# outcome-class samplers in profile_generator.py override certain fields.
# Only schema-level distribution changes for the following fields have effect:
#
#   Always reads from schema (all outcome classes):
#     age, city_tier, employment_type, sector, loan_purpose, loan_tenure_months,
#     credit_vintage_years
#
#   Reads from schema for CONDITIONAL + DECLINE (overridden in APPROVE):
#     monthly_income, loan_amount, employment_tenure_years, annual_interest_rate
#
#   Schema value NEVER used — always set post-sampling or by triggers:
#     cibil_score, missed_payments_24m, settled_accounts_ever, existing_emi_monthly
#
# Distributions marked [realistic] use fitted Indian macro-data.
# Distributions marked [uniform] are either schema-ineffective, categorical,
# or fields where uniform is an acceptable approximation at this scale.

PROFILE_SCHEMA: dict[str, FieldSampler] = {
    # Demographics
    "age":                      UniformInt(22, 65),                    # [uniform] acceptable approximation
    "city_tier":                Choice(["tier1", "tier2", "tier3"]),   # [uniform] categorical

    # Employment
    "employment_type": Choice([
        "salaried_private", "salaried_govt",
        "self_employed_professional", "self_employed_business",
    ]),
    # [realistic] Log-normal: median ₹26.7k, mean ₹35k, p90 ₹65.7k
    # Fitted to NSSO/PLFS urban employed population. ~15% of draws clamp
    # to the ₹15k floor (left tail of log-normal below minimum threshold).
    "monthly_income":           LogNormalSampler(mu=10.2, sigma=0.7, lo=15_000, hi=5_00_000, precision=0),
    # [realistic] Weibull(shape<1): high early-tenure churn consistent with
    # Indian private sector. Median 3.2 yrs, p25 = 1.2 yrs, p75 = 7.1 yrs.
    "employment_tenure_years":  WeibullSampler(shape=0.9, scale=5.0, lo=0.25, hi=25.0, precision=2),
    "sector": Choice([
        "IT", "manufacturing", "healthcare",
        "finance", "retail", "education", "govt",
    ]),

    # Credit profile
    # [uniform — schema-ineffective] always overridden by outcome-class samplers
    "cibil_score":              UniformInt(550, 900),
    # [uniform — schema-ineffective] always set post-sampling as fraction of income
    "existing_emi_monthly":     UniformFloat(0, 80_000, precision=0),
    "loan_accounts_active":     UniformInt(0, 5),                      # [uniform] acceptable
    # [uniform — schema-ineffective] always overridden
    "missed_payments_24m":      UniformInt(0, 6),
    # settled_accounts_ever: loan closed for less than full outstanding amount —
    # lender wrote off part of the debt. Significant negative credit event.
    # [uniform — schema-ineffective] always overridden
    "settled_accounts_ever":    UniformInt(0, 3),
    # [realistic] Weibull(shape>1): slightly peaked, skewed toward 1-5 year
    # histories. Median 2.9 yrs — reflects India's young credit market.
    "credit_vintage_years":     WeibullSampler(shape=1.2, scale=4.0, lo=0.0, hi=15.0, precision=1),

    # Loan request
    # [realistic] Log-normal: median ₹1.48L, p75 ₹2.7L, p90 ₹4.7L.
    # Consistent with RBI personal loan size distribution.
    "loan_amount":              LogNormalSampler(mu=11.9, sigma=0.9, lo=50_000, hi=50_00_000, precision=0),
    "loan_tenure_months":       Choice([12, 24, 36, 48, 60, 84]),      # [uniform] discrete set
    "loan_purpose": Choice([
        "home_renovation", "medical_emergency",
        "education", "debt_consolidation",
        "business_expansion", "wedding",
        "travel", "vehicle_purchase",
    ]),
    "annual_interest_rate":     UniformFloat(10.5, 26.0, precision=2), # [uniform] acceptable
}


# ─── Base profile sampler ─────────────────────────────────────────────────────

def sample_base_profile(
    rng: random.Random,
    overrides: dict[str, FieldSampler] | None = None,
    schema: dict[str, FieldSampler] = PROFILE_SCHEMA,
) -> dict:
    """
    Sample every field from `schema`, substituting `overrides` where provided.

    `overrides` maps field names to FieldSampler instances (not raw values),
    so targeted outcome-class samplers can constrain specific fields while
    inheriting the schema's distribution for everything else.

    Example — pin CIBIL to the approve range, keep everything else as-is:
        p = sample_base_profile(rng, overrides={
            "cibil_score": UniformInt(760, 900),
            "missed_payments_24m": Fixed(0),
        })
    """
    effective = {**schema, **(overrides or {})}
    return {field_name: sampler.sample(rng) for field_name, sampler in effective.items()}
