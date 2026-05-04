"""Deterministic parser for credit memo structured output. Never raises."""
import re
from serving.schemas import CreditDecision, CreditMemo, RiskGrade

_HEADERS = [
    "## APPLICANT SUMMARY",
    "## DEBT SERVICEABILITY",
    "## CREDIT BEHAVIOR",
    "## RISK FLAGS",
    "## RECOMMENDATION",
    "## ANALYST NOTES",
]


def _extract_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for i, header in enumerate(_HEADERS):
        if header not in text:
            continue
        start = text.index(header) + len(header)
        following = [text.find(h, start) for h in _HEADERS[i + 1:] if text.find(h, start) > 0]
        end = min(following) if following else len(text)
        sections[header] = text[start:end].strip()
    return sections


def _extract_decision(rec: str, errors: list[str]) -> CreditDecision:
    m = re.search(r"DECISION:\s*(CONDITIONAL APPROVE|APPROVE|DECLINE)", rec, re.IGNORECASE)
    if not m:
        errors.append("Could not extract DECISION field")
        return CreditDecision.UNKNOWN
    mapping = {
        "APPROVE":            CreditDecision.APPROVE,
        "CONDITIONAL APPROVE": CreditDecision.CONDITIONAL_APPROVE,
        "DECLINE":            CreditDecision.DECLINE,
    }
    return mapping.get(m.group(1).upper(), CreditDecision.UNKNOWN)


def _extract_conditions(rec: str) -> list[str]:
    m = re.search(r"CONDITIONS:(.*?)(?:RISK GRADE:|$)", rec, re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    text = m.group(1).strip()
    if text.lower() in {"none", "n/a", "-", "na"}:
        return []
    # numbered list
    items = re.findall(r"\d+[.)]\s+(.+)", text)
    if items:
        return [i.strip() for i in items]
    # bullet list
    items = re.findall(r"^[-•*]\s+(.+)", text, re.MULTILINE)
    return [i.strip() for i in items]


def _extract_risk_grade(rec: str, errors: list[str]) -> RiskGrade:
    m = re.search(r"RISK GRADE:\s*([ABC][+-]?)", rec, re.IGNORECASE)
    if not m:
        errors.append("Could not extract RISK GRADE field")
        return RiskGrade.UNKNOWN
    grade_map = {
        "A": RiskGrade.A, "B+": RiskGrade.B_PLUS,
        "B": RiskGrade.B, "B-": RiskGrade.B_MINUS,
        "C": RiskGrade.C,
    }
    return grade_map.get(m.group(1).upper(), RiskGrade.UNKNOWN)


def _extract_field(rec: str, label: str, errors: list[str], default: str = "Unknown") -> str:
    m = re.search(rf"{label}:\s*(.+)", rec, re.IGNORECASE)
    if not m:
        errors.append(f"Could not extract {label} field")
        return default
    return m.group(1).strip().split("\n")[0].strip()


def _extract_bullets(section: str) -> list[str]:
    return [b.strip() for b in re.findall(r"^[-•*]\s+(.+)", section, re.MULTILINE)]


def parse(raw_output: str) -> CreditMemo:
    errors: list[str] = []
    sections = _extract_sections(raw_output)

    rec = sections.get("## RECOMMENDATION", "")
    decision = _extract_decision(rec, errors)
    conditions = _extract_conditions(rec)
    risk_grade = _extract_risk_grade(rec, errors)
    authority = _extract_field(rec, "DECISION AUTHORITY", errors)
    review_trigger = _extract_field(rec, "REVIEW TRIGGER", errors)

    flags_section = sections.get("## RISK FLAGS", "")
    risk_flags = _extract_bullets(flags_section)
    if not (2 <= len(risk_flags) <= 4):
        errors.append(f"Risk flags count {len(risk_flags)} outside 2-4 range")

    return CreditMemo(
        applicant_summary=sections.get("## APPLICANT SUMMARY", ""),
        debt_serviceability=sections.get("## DEBT SERVICEABILITY", ""),
        credit_behavior=sections.get("## CREDIT BEHAVIOR", ""),
        risk_flags=risk_flags or ["[PARSE ERROR]"],
        decision=decision,
        conditions=conditions,
        risk_grade=risk_grade,
        decision_authority=authority,
        review_trigger=review_trigger,
        analyst_notes=sections.get("## ANALYST NOTES", ""),
        raw_output=raw_output,
        parse_success=len(errors) == 0,
        parse_errors=errors,
    )
