from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class LoanApplicationCreate(BaseModel):
    full_name: str
    age: int
    annual_income: float
    employment_length: int
    employment_type: str
    home_ownership: str
    existing_debt: float
    num_credit_lines: int
    credit_history_length: int
    late_payments: int
    bankruptcies: int
    loan_amount: float
    loan_purpose: str
    loan_tenure: int

class LoanApplicationResponse(BaseModel):
    id: int
    user_id: int
    status: str
    full_name: str
    age: int
    annual_income: float
    employment_length: int
    employment_type: str
    home_ownership: str
    existing_debt: float
    num_credit_lines: int
    credit_history_length: int
    late_payments: int
    bankruptcies: int
    loan_amount: float
    loan_purpose: str
    loan_tenure: int
    debt_to_income_ratio: float
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
