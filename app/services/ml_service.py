import pickle
import numpy as np
import logging
import json
from app.config import MODEL_PATH, EXPLAINER_PATH, FEATURE_NAMES
from app.models import LoanApplication

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self.model = None
        self.explainer = None
        
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load model from {MODEL_PATH}: {e}")
            
        try:
            with open(EXPLAINER_PATH, 'rb') as f:
                self.explainer = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load explainer from {EXPLAINER_PATH}: {e}")

    def predict(self, application: LoanApplication) -> dict:
        emp_type_map = {'unemployed': 0, 'student': 1, 'retired': 2, 'self_employed': 3, 'salaried': 4}
        home_ownership_map = {'other': 0, 'rent': 1, 'mortgage': 2, 'own': 3}
        loan_purpose_map = {'personal': 0, 'medical': 1, 'education': 2, 'car': 3, 'home': 4, 'business': 5, 'debt_consolidation': 6}

        features = [
            float(application.age),
            float(application.annual_income),
            float(application.employment_length),
            emp_type_map.get(str(application.employment_type).lower(), 0),
            home_ownership_map.get(str(application.home_ownership).lower(), 0),
            float(application.existing_debt),
            float(application.num_credit_lines),
            float(application.credit_history_length),
            float(application.late_payments),
            float(application.bankruptcies),
            float(application.loan_amount),
            loan_purpose_map.get(str(application.loan_purpose).lower(), 0),
            float(application.loan_tenure),
            float(application.debt_to_income_ratio)
        ]

        X = np.array([features])
        
        prob_default = 0.5
        if self.model:
            prob = self.model.predict_proba(X)
            prob_default = prob[0][1] if prob.shape[1] > 1 else prob[0][0]
            
        credit_score = int((1 - prob_default) * 1000)
        
        if credit_score >= 750:
            risk_category = 'low'
        elif credit_score >= 600:
            risk_category = 'medium'
        elif credit_score >= 400:
            risk_category = 'high'
        else:
            risk_category = 'very_high'
            
        shap_values_out = {}
        feature_contributions = []
        top_risk_factors = []
        top_protective_factors = []
        
        if self.explainer:
            try:
                sv = self.explainer.shap_values(X)
                if isinstance(sv, list):
                    shap_vals = sv[1][0] if len(sv) > 1 else sv[0][0]
                elif hasattr(sv, 'values'):
                    shap_vals = sv.values[0]
                    if len(shap_vals.shape) > 1:
                        shap_vals = shap_vals[:, 1] if shap_vals.shape[1] > 1 else shap_vals[:, 0]
                else:
                    shap_vals = sv[0]
                    
                contributions = []
                for i, name in enumerate(self.feature_names):
                    val = shap_vals[i]
                    shap_values_out[name] = float(val)
                    human_name = name.replace('_', ' ').capitalize()
                    feat_val = features[i]
                    direction = 'increases' if val > 0 else 'decreases'
                    impact = 'significantly' if abs(val) > 0.1 else 'slightly'
                    contributions.append({
                        'feature': name,
                        'human_name': human_name,
                        'value': feat_val,
                        'shap': float(val),
                        'impact': impact,
                        'direction': direction
                    })
                    
                feature_contributions = sorted(contributions, key=lambda x: abs(x['shap']), reverse=True)
                
                sorted_by_val = sorted(contributions, key=lambda x: x['shap'])
                top_protective = sorted_by_val[:3]
                top_risk = sorted_by_val[-3:]
                top_risk.reverse()
                
                for f in top_risk:
                    if f['shap'] > 0:
                        top_risk_factors.append(f"Your {f['human_name'].lower()} ({f['value']}) {f['impact']} increases your default risk")
                
                for f in top_protective:
                    if f['shap'] < 0:
                        top_protective_factors.append(f"Your {f['human_name'].lower()} ({f['value']}) {f['impact']} decreases your default risk")

            except Exception as e:
                logger.warning(f"Error computing SHAP values: {e}")
                
        rec_summaries = {
            'low': 'Excellent credit profile. You are highly likely to be approved for most loan products.',
            'medium': 'Good credit profile. You are likely to be approved, but may face slightly higher interest rates.',
            'high': 'High risk profile. Approval may be difficult, consider improving your credit factors before applying.',
            'very_high': 'Very high risk profile. It is strongly recommended to work on reducing debt and improving payment history.'
        }
        
        return {
            'credit_score': credit_score,
            'risk_category': risk_category,
            'probability_of_default': float(prob_default),
            'shap_values': json.dumps(shap_values_out),
            'feature_contributions': json.dumps(feature_contributions),
            'top_risk_factors': json.dumps(top_risk_factors),
            'top_protective_factors': json.dumps(top_protective_factors),
            'recommendation_summary': rec_summaries.get(risk_category, '')
        }

ml_service = MLService()
