from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base

class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(Integer, ForeignKey("loan_applications.id"), unique=True)
    credit_score = Column(Integer)
    risk_category = Column(String(50))
    probability_of_default = Column(Float)
    shap_values = Column(Text)
    feature_contributions = Column(Text)
    top_risk_factors = Column(Text)
    top_protective_factors = Column(Text)
    recommendation_summary = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    application = relationship("LoanApplication", back_populates="assessment")
