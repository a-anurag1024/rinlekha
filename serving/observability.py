"""
Langfuse observability wrapper around the serving pipeline.

Every call to generate_credit_memo_observed() is traced with:
  - Input: full borrower profile dict
  - Output: structured CreditMemo (minus raw_output to keep traces compact)
  - Scores: parse_success, structural_compliance, decision_extracted
  - Metadata: cibil_band, foir_band, employment_type, loan_purpose

Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in environment.
Compatible with Langfuse v4.x (get_client / score_current_span API).
"""
from dotenv import load_dotenv
load_dotenv()  # must run before get_client() reads env vars

from langfuse import get_client, observe

from serving.pipeline import generate_credit_memo
from serving.schemas import CreditDecision, CreditMemo

langfuse = get_client()

_SECTION_HEADERS = [
    "## APPLICANT SUMMARY", "## DEBT SERVICEABILITY",
    "## CREDIT BEHAVIOR",   "## RISK FLAGS",
    "## RECOMMENDATION",    "## ANALYST NOTES",
]


def _structural_score(raw: str) -> float:
    """Fast rule-based score — same logic as StructuralComplianceMetric."""
    checks = [h in raw for h in _SECTION_HEADERS]
    positions = [raw.find(h) for h in _SECTION_HEADERS if h in raw]
    checks.append(positions == sorted(positions))
    for i, h in enumerate(_SECTION_HEADERS):
        if h in raw:
            start = raw.index(h) + len(h)
            rest = [raw.find(s, start) for s in _SECTION_HEADERS if raw.find(s, start) > 0]
            end = min(rest) if rest else len(raw)
            checks.append(len(raw[start:end].strip()) > 30)
    return sum(checks) / len(checks) if checks else 0.0


def _cibil_band(score: int) -> str:
    if score == 0:      return "no_history"
    if score < 620:     return "subprime"
    if score < 700:     return "near_prime"
    if score < 750:     return "prime"
    return "superprime"


def _foir_band(foir_pct: float) -> str:
    if foir_pct < 30:   return "low"
    if foir_pct < 45:   return "moderate"
    if foir_pct < 55:   return "near_ceiling"
    return "over_ceiling"


@observe(name="generate_credit_memo")
def generate_credit_memo_observed(profile: dict) -> CreditMemo:
    cibil = profile.get("cibil_score", 0)
    income = profile.get("monthly_income", 1)
    existing_emi = profile.get("existing_emi_monthly", 0)
    proposed_emi = profile.get("proposed_emi_monthly", 0)
    foir_post = (existing_emi + proposed_emi) / income * 100 if income else 0

    langfuse.update_current_span(
        input=profile,
        metadata={
            "cibil_score":     cibil,
            "cibil_band":      _cibil_band(cibil),
            "foir_post_loan":  round(foir_post, 1),
            "foir_band":       _foir_band(foir_post),
            "employment_type": profile.get("employment_type"),
            "loan_purpose":    profile.get("loan_purpose"),
            "model":           "rinlekha-gemma3-4b-finetuned",
        },
    )

    memo: CreditMemo = generate_credit_memo(profile)

    langfuse.score_current_span(
        name="parse_success",
        value=1.0 if memo.parse_success else 0.0,
        comment="; ".join(memo.parse_errors) if memo.parse_errors else None,
    )
    langfuse.score_current_span(
        name="structural_compliance",
        value=_structural_score(memo.raw_output),
    )
    langfuse.score_current_span(
        name="decision_extracted",
        value=0.0 if memo.decision == CreditDecision.UNKNOWN else 1.0,
    )

    langfuse.update_current_span(
        output=memo.model_dump(exclude={"raw_output"}),
    )

    return memo
