from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base

class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String(50), default='pending')
    full_name = Column(String(255))
    age = Column(Integer)
    annual_income = Column(Float)
    employment_length = Column(Integer)
    employment_type = Column(String(50))
    home_ownership = Column(String(50))
    existing_debt = Column(Float)
    num_credit_lines = Column(Integer)
    credit_history_length = Column(Integer)
    late_payments = Column(Integer)
    bankruptcies = Column(Integer)
    loan_amount = Column(Float)
    loan_purpose = Column(String(100))
    loan_tenure = Column(Integer)
    debt_to_income_ratio = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="applications")
    assessment = relationship("RiskAssessment", back_populates="application", uselist=False)
