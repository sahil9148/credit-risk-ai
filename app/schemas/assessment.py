from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class RiskAssessmentResponse(BaseModel):
    id: int
    application_id: int
    credit_score: int
    risk_category: str
    probability_of_default: float
    shap_values: str
    feature_contributions: str
    top_risk_factors: str
    top_protective_factors: str
    recommendation_summary: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
