"""Unit tests for serving/parser.py — no server required."""
import pytest
from serving.parser import parse
from serving.schemas import CreditDecision, RiskGrade

GOOD_MEMO = """\
## APPLICANT SUMMARY
34-year-old salaried private employee in IT sector with 4.5 years tenure.

## DEBT SERVICEABILITY
Post-loan FOIR of 38.2% is within the 55% policy ceiling. Proposed EMI of ₹18,450 appears manageable.

## CREDIT BEHAVIOR
CIBIL score of 724 indicates good credit standing. One missed payment in last 24 months is noted.

## RISK FLAGS
- Single missed payment in last 24 months may indicate transient stress.
- Moderate active loan count of 2 warrants monitoring.

## RECOMMENDATION
DECISION: CONDITIONAL APPROVE
CONDITIONS:
1. Last 6 months salary slips
2. Form 16 for last 2 years
RISK GRADE: B
DECISION AUTHORITY: Branch Credit Manager
REVIEW TRIGGER: Any further missed payment within 12 months

## ANALYST NOTES
Profile is borderline positive. Conditions address the delinquency concern.
"""

DECLINE_MEMO = """\
## APPLICANT SUMMARY
Borrower summary text here.

## DEBT SERVICEABILITY
Post-loan FOIR exceeds policy ceiling at 87%.

## CREDIT BEHAVIOR
CIBIL 820 is strong but FOIR is disqualifying.

## RISK FLAGS
- Post-loan FOIR of 87% is far above the 55% policy ceiling.
- Existing EMI burden already elevated pre-loan.
- Self-employed income may fluctuate under economic stress.

## RECOMMENDATION
DECISION: DECLINE
CONDITIONS: N/A
RISK GRADE: C
DECISION AUTHORITY: Regional Credit Committee
REVIEW TRIGGER: Significant reduction in existing loan burden

## ANALYST NOTES
Despite strong CIBIL, debt serviceability cannot be established.
"""


def test_parse_conditional_approve():
    memo = parse(GOOD_MEMO)
    assert memo.parse_success
    assert memo.decision == CreditDecision.CONDITIONAL_APPROVE
    assert memo.risk_grade == RiskGrade.B
    assert len(memo.risk_flags) == 2
    assert len(memo.conditions) == 2
    assert "salary slips" in memo.conditions[0].lower()
    assert memo.decision_authority == "Branch Credit Manager"
    assert "APPLICANT SUMMARY" not in memo.applicant_summary


def test_parse_decline():
    memo = parse(DECLINE_MEMO)
    assert memo.parse_success  # 3 flags is within 2-4 range
    assert memo.decision == CreditDecision.DECLINE
    assert memo.risk_grade == RiskGrade.C
    assert memo.conditions == []
    assert len(memo.risk_flags) == 3


def test_parse_empty():
    memo = parse("")
    assert not memo.parse_success
    assert memo.decision == CreditDecision.UNKNOWN
    assert memo.risk_grade == RiskGrade.UNKNOWN
    assert len(memo.parse_errors) > 0


def test_parse_missing_decision():
    broken = GOOD_MEMO.replace("DECISION: CONDITIONAL APPROVE", "")
    memo = parse(broken)
    assert not memo.parse_success
    assert memo.decision == CreditDecision.UNKNOWN
    assert any("DECISION" in e for e in memo.parse_errors)


def test_parse_approve_no_conditions():
    approve_memo = GOOD_MEMO.replace("DECISION: CONDITIONAL APPROVE", "DECISION: APPROVE")
    memo = parse(approve_memo)
    assert memo.decision == CreditDecision.APPROVE


def test_raw_output_preserved():
    memo = parse(GOOD_MEMO)
    assert memo.raw_output == GOOD_MEMO
