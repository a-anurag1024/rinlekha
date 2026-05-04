"""Pydantic output schema for structured credit memo objects."""
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class CreditDecision(str, Enum):
    APPROVE = "APPROVE"
    CONDITIONAL_APPROVE = "CONDITIONAL APPROVE"
    DECLINE = "DECLINE"
    UNKNOWN = "UNKNOWN"


class RiskGrade(str, Enum):
    A = "A"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C = "C"
    UNKNOWN = "UNKNOWN"


class CreditMemo(BaseModel):
    applicant_summary:   str = Field(default="")
    debt_serviceability: str = Field(default="")
    credit_behavior:     str = Field(default="")
    risk_flags:          list[str] = Field(default_factory=list)
    decision:            CreditDecision = CreditDecision.UNKNOWN
    conditions:          list[str] = Field(default_factory=list)
    risk_grade:          RiskGrade = RiskGrade.UNKNOWN
    decision_authority:  str = Field(default="Unknown")
    review_trigger:      str = Field(default="Unknown")
    analyst_notes:       str = Field(default="")
    raw_output:          str = Field(default="")
    parse_success:       bool = False
    parse_errors:        list[str] = Field(default_factory=list)

    @field_validator("risk_flags")
    @classmethod
    def validate_flag_count(cls, v):
        # validator is informational only — parse errors capture violations
        return v