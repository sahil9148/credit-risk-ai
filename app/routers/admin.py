from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models import User, LoanApplication, RiskAssessment
from app.services.auth_service import get_admin_user

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.get("/stats")
def get_admin_stats(admin_user: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    total_users = db.query(User).count()
    total_applications = db.query(LoanApplication).count()
    
    avg_score = db.query(func.avg(RiskAssessment.credit_score)).scalar() or 0
    
    approved = db.query(LoanApplication).filter(LoanApplication.status == "approved").count()
    approval_rate = (approved / total_applications * 100) if total_applications > 0 else 0
    
    risk_dist = db.query(RiskAssessment.risk_category, func.count(RiskAssessment.id)).group_by(RiskAssessment.risk_category).all()
    risk_distribution = {cat: count for cat, count in risk_dist}
    
    recent = db.query(LoanApplication).order_by(LoanApplication.created_at.desc()).limit(10).all()
    recent_apps = []
    for app in recent:
        score = None
        if app.assessment:
            score = app.assessment.credit_score
        recent_apps.append({
            "id": app.id,
            "full_name": app.full_name,
            "status": app.status,
            "score": score
        })
        
    return {
        "total_users": total_users,
        "total_applications": total_applications,
        "avg_credit_score": round(avg_score, 1),
        "approval_rate": round(approval_rate, 1),
        "risk_distribution": risk_distribution,
        "recent_applications": recent_apps,
        "monthly_applications": {}
    }

from pydantic import BaseModel

class StatusUpdate(BaseModel):
    status: str

@router.post("/applications/{id}/status")
def update_status(id: int, body: StatusUpdate, admin_user: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    app = db.query(LoanApplication).filter(LoanApplication.id == id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Not found")
    app.status = body.status
    db.commit()
    return {"message": "Status updated"}
