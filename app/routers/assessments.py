from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.services.ml_service import ml_service
from app.models import LoanApplication
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/api/assessments", tags=["assessments"])

class WhatIfRequest(BaseModel):
    age: int
    annual_income: float
    employment_length: float
    employment_type: str
    home_ownership: str
    existing_debt: float
    num_credit_lines: int
    credit_history_length: float
    late_payments: int
    bankruptcies: int
    loan_amount: float
    loan_purpose: str
    loan_tenure: int

@router.post("/whatif")
def what_if_analysis(req: WhatIfRequest, current_user = Depends(get_current_user)):
    dti_ratio = req.existing_debt / req.annual_income if req.annual_income > 0 else 0.0
    
    dummy_app = LoanApplication(
        age=req.age,
        annual_income=req.annual_income,
        employment_length=req.employment_length,
        employment_type=req.employment_type,
        home_ownership=req.home_ownership,
        existing_debt=req.existing_debt,
        num_credit_lines=req.num_credit_lines,
        credit_history_length=req.credit_history_length,
        late_payments=req.late_payments,
        bankruptcies=req.bankruptcies,
        loan_amount=req.loan_amount,
        loan_purpose=req.loan_purpose,
        loan_tenure=req.loan_tenure,
        debt_to_income_ratio=dti_ratio
    )
    
    result = ml_service.predict(dummy_app)
    return result
