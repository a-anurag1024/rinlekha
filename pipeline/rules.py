"""
pipeline/rules.py — Domain-specific rule registries for RinLekha.

All NBFC underwriting knowledge lives here.  The three generator functions
in profile_generator.py (compute_derived_fields, determine_outcome,
_sample_decline) contain zero domain logic — they just iterate the
registries defined below.

EXTENSION POINTS
────────────────
Add / change a derived field    → edit DERIVED_FIELD_RULES
Change a policy threshold       → edit the predicate lambda in HARD_DECLINE_RULES
Add a new hard decline reason   → append a HardDeclineRule
Add a new conditional condition → append a ConditionalRule
Change approve criteria         → edit CLEAN_APPROVE_RULE.predicates
Add a new decline trigger       → append a DeclineTrigger with compatible_with set

RULE ORDERING
─────────────
DERIVED_FIELD_RULES   — applied in list order; later rules may use fields
                        added by earlier ones (e.g. proposed_emi must come
                        before foir_post_loan).
HARD_DECLINE_RULES    — all triggered; ordering only matters for readability.
CONDITIONAL_RULES     — all triggered; ordering determines condition list order
                        in the output (affects memo narrative quality).
DECLINE_TRIGGERS      — each declares its own compatible_with list; the
                        pairing matrix is derived automatically.

NOTE ON DEPENDENT FIELD SAMPLING
─────────────────────────────────
Some fields cannot be sampled independently because their realistic range
depends on another field (e.g. existing_emi_monthly should not exceed ~40%
of monthly_income, else derived FOIR is nonsensical before a loan is added).
These are handled as post-sampling adjustments directly in the sampler
functions in profile_generator.py rather than here, because they require
access to the already-sampled profile state.

A future extension could introduce a ContextualFieldSampler protocol:
    def sample(self, rng, context: dict) -> Any: ...
where context is the partially-built profile.  That would make the
dependency graph explicit and allow everything to live in samplers.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable


# ─── Rule dataclasses ─────────────────────────────────────────────────────────

@dataclass
class DerivedFieldRule:
    """
    Adds one computed field to the profile dict.
    `compute` receives the full profile (with all previously added fields)
    and returns the value for `name`.
    """
    name: str
    compute: Callable[[dict], Any]
    doc: str = ""


@dataclass
class HardDeclineRule:
    """
    One hard-decline condition.  If predicate(profile) is True, condition_key
    is appended to the DECLINE reasons list.  All triggered rules fire.
    """
    condition_key: str
    predicate: Callable[[dict], bool]
    doc: str = ""


@dataclass
class ApproveRule:
    """
    Clean APPROVE: all predicates must hold and no HardDeclineRule triggered.
    Predicates are listed individually so each criterion is visible on its own.
    """
    predicates: list[Callable[[dict], bool]]
    doc: str = ""

    def satisfied(self, profile: dict) -> bool:
        return all(pred(profile) for pred in self.predicates)


@dataclass
class ConditionalRule:
    """
    One condition appended to a CONDITIONAL_APPROVE outcome.
    All triggered ConditionalRules contribute to the conditions list.
    """
    condition_key: str
    predicate: Callable[[dict], bool]
    doc: str = ""


@dataclass
class DeclineTrigger:
    """
    A field-injection strategy used by _sample_decline to produce profiles
    that exercise a specific hard-decline condition.

    apply(profile, rng) — mutates profile in-place, returns it.
    compatible_with     — names of other triggers that can co-occur in the
                          same profile (used to build the pairing matrix
                          automatically; no manual list elsewhere).
    """
    name: str
    apply: Callable[[dict, random.Random], dict]
    compatible_with: list[str] = field(default_factory=list)
    doc: str = ""


# ─── Derived field rules ──────────────────────────────────────────────────────
# Applied in order — later rules may reference fields set by earlier ones.

def _emi(p: dict) -> float:
    r = p["annual_interest_rate"] / 12 / 100
    n = p["loan_tenure_months"]
    principal = p["loan_amount"]
    # Standard reducing-balance EMI formula (used by all Indian banks/NBFCs).
    # Keeps EMI constant; interest component shrinks and principal grows monthly.
    return round(principal * r * (1 + r) ** n / ((1 + r) ** n - 1), 2)


DERIVED_FIELD_RULES: list[DerivedFieldRule] = [
    DerivedFieldRule(
        name="proposed_emi",
        compute=_emi,
        doc="Monthly repayment on the requested loan.",
    ),
    DerivedFieldRule(
        name="foir_pre_loan",
        compute=lambda p: round(p["existing_emi_monthly"] / p["monthly_income"], 3),
        doc="Fixed Obligation to Income Ratio before the proposed loan.",
    ),
    DerivedFieldRule(
        name="foir_post_loan",
        compute=lambda p: round(
            (p["existing_emi_monthly"] + p["proposed_emi"]) / p["monthly_income"], 3
        ),
        doc="FOIR after adding proposed EMI — primary serviceability metric.",
    ),
    DerivedFieldRule(
        name="loan_to_income_ratio",
        compute=lambda p: round(p["loan_amount"] / p["monthly_income"], 1),
        doc="Loan principal as a multiple of monthly income.",
    ),
]


# ─── Hard decline rules ───────────────────────────────────────────────────────

HARD_DECLINE_RULES: list[HardDeclineRule] = [
    HardDeclineRule(
        condition_key="credit_score_below_minimum_threshold",
        predicate=lambda p: p["cibil_score"] < 620,
        doc="CIBIL below 620 — minimum floor for NBFC consideration.",
    ),
    HardDeclineRule(
        condition_key="foir_exceeds_policy_ceiling_of_55pct",
        predicate=lambda p: p["foir_post_loan"] > 0.55,
        doc="Post-loan FOIR > 55% — borrower cannot service the additional EMI.",
    ),
    HardDeclineRule(
        condition_key="excessive_recent_delinquency",
        predicate=lambda p: p["missed_payments_24m"] >= 4,
        doc="4+ missed payments in 24 months — pattern of non-payment.",
    ),
    HardDeclineRule(
        condition_key="multiple_settled_loan_accounts",
        predicate=lambda p: p["settled_accounts_ever"] >= 2,
        doc="2+ settled accounts — repeated inability to honour full obligations.",
    ),
    HardDeclineRule(
        condition_key="loan_amount_grossly_disproportionate_to_income",
        predicate=lambda p: p["loan_to_income_ratio"] > 60,
        doc="Loan > 60× monthly income — structurally unrepayable.",
    ),
]


# ─── Clean approve rule ───────────────────────────────────────────────────────

CLEAN_APPROVE_RULE = ApproveRule(
    predicates=[
        lambda p: p["cibil_score"] >= 760,
        lambda p: p["foir_post_loan"] <= 0.38,
        lambda p: p["missed_payments_24m"] == 0,
        lambda p: p["employment_tenure_years"] >= 2,
        lambda p: p["settled_accounts_ever"] == 0,
    ],
    doc="All five criteria must hold simultaneously for a clean approve.",
)


# ─── Conditional requirement rules ───────────────────────────────────────────

CONDITIONAL_RULES: list[ConditionalRule] = [
    ConditionalRule(
        condition_key="income_proof_last_6_months_salary_slips",
        predicate=lambda p: p["foir_post_loan"] > 0.46,
        doc="FOIR 46–55%: marginal serviceability — verify income stability.",
    ),
    ConditionalRule(
        condition_key="written_explanation_for_missed_payments",
        predicate=lambda p: p["missed_payments_24m"] in [1, 2],
        doc="1–2 missed payments: may be situational — seek written explanation.",
    ),
    ConditionalRule(
        condition_key="employment_confirmation_from_hr",
        predicate=lambda p: p["employment_tenure_years"] < 1.0,
        doc="< 1 year tenure: income continuity risk — confirm employment.",
    ),
    ConditionalRule(
        condition_key="additional_income_proof_or_co_applicant",
        predicate=lambda p: p["loan_to_income_ratio"] > 30,
        doc="Loan > 30× income: needs co-applicant or additional income proof.",
    ),
    ConditionalRule(
        condition_key="guarantor_or_collateral_security",
        predicate=lambda p: p["cibil_score"] < 680,
        doc="CIBIL 620–679: weak credit — requires guarantor or collateral.",
    ),
    ConditionalRule(
        condition_key="noc_from_previous_lender_for_settled_account",
        predicate=lambda p: p["settled_accounts_ever"] == 1,
        doc="1 settled account: one-off may be acceptable — NOC required.",
    ),
]

# Fallback condition when no specific rule triggers (profile is marginal but clean)
CONDITIONAL_FALLBACK = "standard_documentation_verification"


# ─── Decline trigger definitions ──────────────────────────────────────────────
# compatible_with declares which other triggers can co-occur in the same profile.
# Pairs are derived automatically — no separate list to maintain.

def _apply_low_cibil(p: dict, rng: random.Random) -> dict:
    p["cibil_score"] = rng.randint(550, 619)
    p.setdefault("loan_amount", round(rng.uniform(50_000, 20_00_000)))
    return p


def _apply_high_foir(p: dict, rng: random.Random) -> dict:
    p["monthly_income"]       = round(rng.uniform(15_000, 80_000))
    p["existing_emi_monthly"] = round(rng.uniform(
        p["monthly_income"] * 0.20,
        min(80_000, p["monthly_income"] * 0.45),
    ))
    r        = p["annual_interest_rate"] / 12 / 100
    n        = p["loan_tenure_months"]
    min_emi  = max(p["monthly_income"] * 0.56 - p["existing_emi_monthly"], 1_000)
    min_loan = min_emi * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
    p["loan_amount"] = round(min(max(min_loan * 1.10, 50_000), 50_00_000))
    return p


def _apply_delinquency(p: dict, rng: random.Random) -> dict:
    p["missed_payments_24m"] = rng.randint(4, 6)
    p.setdefault("loan_amount", round(rng.uniform(50_000, 20_00_000)))
    return p


def _apply_settled(p: dict, rng: random.Random) -> dict:
    p["settled_accounts_ever"] = rng.randint(2, 3)
    p.setdefault("loan_amount", round(rng.uniform(50_000, 20_00_000)))
    return p


def _apply_high_lti(p: dict, rng: random.Random) -> dict:
    p["monthly_income"] = round(rng.uniform(15_000, 60_000))
    min_loan = p["monthly_income"] * 61
    p["loan_amount"] = round(rng.uniform(min_loan, min(min_loan * 2, 50_00_000)))
    return p


DECLINE_TRIGGERS: list[DeclineTrigger] = [
    DeclineTrigger(
        name="low_cibil",
        apply=_apply_low_cibil,
        compatible_with=["delinquency", "settled", "high_foir"],
        doc="CIBIL 550–619 — below policy minimum.",
    ),
    DeclineTrigger(
        name="high_foir",
        apply=_apply_high_foir,
        compatible_with=["low_cibil", "high_lti"],
        doc="Post-loan FOIR > 55% — over-leveraged relative to income.",
    ),
    DeclineTrigger(
        name="delinquency",
        apply=_apply_delinquency,
        compatible_with=["low_cibil", "settled"],
        doc="4–6 missed payments in 24 months — serial delinquency.",
    ),
    DeclineTrigger(
        name="settled",
        apply=_apply_settled,
        compatible_with=["low_cibil", "delinquency"],
        doc="2–3 settled accounts — repeated debt write-offs.",
    ),
    DeclineTrigger(
        name="high_lti",
        apply=_apply_high_lti,
        compatible_with=["high_foir"],
        doc="Loan > 60× monthly income — structurally unrepayable.",
    ),
]


# ─── Derived compatibility matrix (auto-built — do not edit manually) ─────────

def build_compatible_pairs(
    triggers: list[DeclineTrigger],
) -> list[tuple[DeclineTrigger, DeclineTrigger]]:
    """
    Derive all valid ordered (t1, t2) pairs from each trigger's compatible_with.
    Uses canonical (name_a < name_b) ordering to avoid duplicates.
    Called once at import time; stored in COMPATIBLE_TRIGGER_PAIRS.
    """
    index = {t.name: t for t in triggers}
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[DeclineTrigger, DeclineTrigger]] = []

    for t in triggers:
        for other_name in t.compatible_with:
            key = tuple(sorted([t.name, other_name]))
            if key not in seen and other_name in index:
                seen.add(key)
                pairs.append((index[key[0]], index[key[1]]))

    return pairs


COMPATIBLE_TRIGGER_PAIRS: list[tuple[DeclineTrigger, DeclineTrigger]] = (
    build_compatible_pairs(DECLINE_TRIGGERS)
)
