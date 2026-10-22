import os
import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import LoanApplication, User
from app.services.auth_service import get_current_user, get_current_user_optional, get_admin_user
from app.services.recommendation import get_loan_recommendations, get_improvement_tips

router = APIRouter(tags=["pages"])

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))


def render(request, name, ctx=None):
    """Helper to render templates compatible with both old and new Starlette."""
    context = ctx or {}
    context["request"] = request
    return templates.TemplateResponse(name=name, context=context, request=request)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, user: User = Depends(get_current_user_optional)):
    return render(request, "index.html", {"current_user": user})


@router.get("/login", response_class=HTMLResponse)
async def login(request: Request, error: str = None):
    return render(request, "auth/login.html", {"error": error})


@router.get("/register", response_class=HTMLResponse)
async def register(request: Request, error: str = None):
    return render(request, "auth/register.html", {"error": error})


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    applications = db.query(LoanApplication).filter(LoanApplication.user_id == current_user.id).order_by(LoanApplication.created_at.desc()).all()
    
    latest_score = None
    latest_risk_category = None
    approved_count = 0
    for app in applications:
        if app.assessment and latest_score is None:
            latest_score = app.assessment.credit_score
            latest_risk_category = app.assessment.risk_category
        if app.status == "approved":
            approved_count += 1
    
    stats = {
        "total_applications": len(applications),
        "latest_score": latest_score,
        "latest_risk_category": latest_risk_category,
        "approval_rate": round(approved_count / len(applications) * 100, 1) if applications else 0
    }
    
    return render(request, "dashboard/index.html", {
        "user": current_user,
        "current_user": current_user,
        "applications": applications,
        "stats": stats
    })


@router.get("/dashboard/apply", response_class=HTMLResponse)
async def apply(request: Request, current_user: User = Depends(get_current_user)):
    return render(request, "dashboard/apply.html", {
        "user": current_user,
        "current_user": current_user
    })


@router.get("/dashboard/report/{application_id}", response_class=HTMLResponse)
async def report(request: Request, application_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    application = db.query(LoanApplication).filter(
        LoanApplication.id == application_id, 
        LoanApplication.user_id == current_user.id
    ).first()
    if not application:
        return RedirectResponse("/dashboard")
        
    assessment = application.assessment
    shap_data = {}
    feature_contributions = []
    risk_factors = []
    protective_factors = []
    recommendations = []
    tips = []
    
    if assessment:
        shap_data = json.loads(assessment.shap_values) if assessment.shap_values else {}
        feature_contributions = json.loads(assessment.feature_contributions) if assessment.feature_contributions else []
        risk_factors = json.loads(assessment.top_risk_factors) if assessment.top_risk_factors else []
        protective_factors = json.loads(assessment.top_protective_factors) if assessment.top_protective_factors else []
        
        recommendations = get_loan_recommendations(
            credit_score=assessment.credit_score,
            loan_amount=application.loan_amount,
            annual_income=application.annual_income,
            loan_purpose=application.loan_purpose
        )
        tips = get_improvement_tips(assessment.credit_score, risk_factors)
        
    return render(request, "dashboard/report.html", {
        "user": current_user,
        "current_user": current_user,
        "application": application,
        "assessment": assessment,
        "shap_data": shap_data,
        "feature_contributions": feature_contributions,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "recommendations": recommendations,
        "improvement_tips": tips
    })


@router.get("/dashboard/recommendations/{application_id}", response_class=HTMLResponse)
async def recommendations_page(request: Request, application_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    application = db.query(LoanApplication).filter(
        LoanApplication.id == application_id, 
        LoanApplication.user_id == current_user.id
    ).first()
    if not application or not application.assessment:
        return RedirectResponse("/dashboard")
        
    recommendations = get_loan_recommendations(
        credit_score=application.assessment.credit_score,
        loan_amount=application.loan_amount,
        annual_income=application.annual_income,
        loan_purpose=application.loan_purpose
    )
    
    return render(request, "dashboard/recommendations.html", {
        "user": current_user,
        "current_user": current_user,
        "application": application,
        "assessment": application.assessment,
        "recommendations": recommendations
    })


@router.get("/dashboard/history", response_class=HTMLResponse)
async def history(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    applications = db.query(LoanApplication).filter(
        LoanApplication.user_id == current_user.id
    ).order_by(LoanApplication.created_at.desc()).all()
    return render(request, "dashboard/history.html", {
        "user": current_user,
        "current_user": current_user,
        "applications": applications
    })


@router.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, current_user: User = Depends(get_admin_user)):
    return render(request, "admin/index.html", {
        "user": current_user,
        "current_user": current_user
    })
