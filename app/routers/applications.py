from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import LoanApplication, RiskAssessment
from app.services.auth_service import get_current_user
from app.services.ml_service import ml_service

router = APIRouter(prefix="/api/applications", tags=["applications"])

@router.post("/")
async def create_application(
    full_name: str = Form(...),
    age: int = Form(...),
    annual_income: float = Form(...),
    employment_length: float = Form(...),
    employment_type: str = Form(...),
    home_ownership: str = Form(...),
    existing_debt: float = Form(...),
    num_credit_lines: int = Form(...),
    credit_history_length: float = Form(...),
    late_payments: int = Form(...),
    bankruptcies: int = Form(...),
    loan_amount: float = Form(...),
    loan_purpose: str = Form(...),
    loan_tenure: int = Form(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dti_ratio = existing_debt / annual_income if annual_income > 0 else 0.0
    
    application = LoanApplication(
        user_id=current_user.id,
        full_name=full_name,
        age=age,
        annual_income=annual_income,
        employment_length=employment_length,
        employment_type=employment_type,
        home_ownership=home_ownership,
        existing_debt=existing_debt,
        num_credit_lines=num_credit_lines,
        credit_history_length=credit_history_length,
        late_payments=late_payments,
        bankruptcies=bankruptcies,
        loan_amount=loan_amount,
        loan_purpose=loan_purpose,
        loan_tenure=loan_tenure,
        debt_to_income_ratio=dti_ratio,
        status="pending"
    )
    
    db.add(application)
    db.commit()
    db.refresh(application)
    
    ml_result = ml_service.predict(application)
    
    assessment = RiskAssessment(
        application_id=application.id,
        credit_score=ml_result['credit_score'],
        risk_category=ml_result['risk_category'],
        probability_of_default=ml_result['probability_of_default'],
        shap_values=ml_result['shap_values'],
        feature_contributions=ml_result['feature_contributions'],
        top_risk_factors=ml_result['top_risk_factors'],
        top_protective_factors=ml_result['top_protective_factors'],
        recommendation_summary=ml_result['recommendation_summary']
    )
    
    db.add(assessment)
    db.commit()
    
    return RedirectResponse(
        url=f"/dashboard/report/{application.id}",
        status_code=status.HTTP_303_SEE_OTHER
    )

@router.get("/{id}")
def get_application(id: int, current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    app = db.query(LoanApplication).filter(LoanApplication.id == id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Not found")
    if app.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    return app

@router.get("/{id}/assessment")
def get_application_assessment(id: int, current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    app = db.query(LoanApplication).filter(LoanApplication.id == id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Not found")
    if app.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    assessment = db.query(RiskAssessment).filter(RiskAssessment.application_id == id).first()
    return assessment
