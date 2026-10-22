import os
import math
import json
import logging
from app.config import MODEL_JSON_PATH, MODEL_PATH, FEATURE_NAMES
from app.models import LoanApplication

logger = logging.getLogger(__name__)

# Lightweight standalone tree predictor (zero heavy dependencies for serverless deployment)
class LightweightTreeModel:
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        self.trees = d['learner']['gradient_booster']['model']['trees']
        raw_bs = d['learner']['learner_model_param']['base_score'].strip('[]')
        self.base_score = float(raw_bs)
        self.base_margin = math.log(self.base_score / (1.0 - self.base_score))
        self.feature_names = d['learner']['feature_names']
        
    def predict_proba(self, features: list) -> float:
        total_margin = 0.0
        for tree in self.trees:
            lefts = tree['left_children']
            rights = tree['right_children']
            split_indices = tree['split_indices']
            split_conditions = tree['split_conditions']
            node = 0
            while lefts[node] != -1:
                feat_idx = split_indices[node]
                val = features[feat_idx]
                if val < split_conditions[node]:
                    node = lefts[node]
                else:
                    node = rights[node]
            total_margin += split_conditions[node]
        margin = total_margin + self.base_margin
        prob = 1.0 / (1.0 + math.exp(-margin))
        return float(prob)
        
    def predict_attributions(self, features: list) -> dict:
        attribs = {name: 0.0 for name in self.feature_names}
        for tree in self.trees:
            lefts = tree['left_children']
            rights = tree['right_children']
            split_indices = tree['split_indices']
            split_conditions = tree['split_conditions']
            base_weights = tree['base_weights']
            node = 0
            while lefts[node] != -1:
                feat_idx = split_indices[node]
                val = features[feat_idx]
                parent_weight = base_weights[node]
                if val < split_conditions[node]:
                    child = lefts[node]
                else:
                    child = rights[node]
                child_weight = base_weights[child]
                attribs[self.feature_names[feat_idx]] += (child_weight - parent_weight)
                node = child
        # Normalize scale for clear visualization on waterfall chart
        max_abs = max([abs(v) for v in attribs.values()] or [1.0])
        return {k: round(v / (max_abs * 2.0), 4) for k, v in attribs.items()}


class MLService:
    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self.lightweight_model = None
        self.xgb_booster = None

        # 1. Preferred: Lightweight pure-Python model (0 MB extra dependencies, works in any serverless lambda)
        if os.path.exists(MODEL_JSON_PATH):
            try:
                self.lightweight_model = LightweightTreeModel(MODEL_JSON_PATH)
                logger.info("Loaded LightweightTreeModel from model.json")
            except Exception as e:
                logger.warning(f"Could not load lightweight tree model: {e}")

        # 2. Optional: Try loading XGBoost booster if xgboost is installed in environment
        try:
            import xgboost as xgb
            if os.path.exists(MODEL_JSON_PATH):
                self.xgb_booster = xgb.Booster()
                self.xgb_booster.load_model(MODEL_JSON_PATH)
        except Exception:
            pass

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

        prob_default = 0.5
        shap_values_out = {}

        # 1. Prediction & Attribution via Lightweight Pure-Python Model
        if self.lightweight_model is not None:
            try:
                prob_default = self.lightweight_model.predict_proba(features)
                shap_values_out = self.lightweight_model.predict_attributions(features)
            except Exception as e:
                logger.warning(f"Lightweight prediction error: {e}")
        # 2. Fallback to XGBoost native booster if available
        elif self.xgb_booster is not None:
            try:
                import xgboost as xgb
                dmat = xgb.DMatrix([features], feature_names=self.feature_names)
                prob_default = float(self.xgb_booster.predict(dmat)[0])
                contribs = self.xgb_booster.predict(dmat, pred_contribs=True)[0]
                shap_values_out = {self.feature_names[i]: float(contribs[i]) for i in range(len(self.feature_names))}
            except Exception as e:
                logger.warning(f"XGBoost booster error: {e}")

        credit_score = int((1 - prob_default) * 1000)
        
        if credit_score >= 750:
            risk_category = 'low'
        elif credit_score >= 600:
            risk_category = 'medium'
        elif credit_score >= 400:
            risk_category = 'high'
        else:
            risk_category = 'very_high'

        feature_contributions = []
        top_risk_factors = []
        top_protective_factors = []

        if shap_values_out:
            contributions = []
            for i, name in enumerate(self.feature_names):
                val = float(shap_values_out.get(name, 0.0))
                human_name = name.replace('_', ' ').capitalize()
                feat_val = features[i]
                direction = 'increases' if val > 0 else 'decreases'
                impact = 'significantly' if abs(val) > 0.1 else 'slightly'
                contributions.append({
                    'feature': name,
                    'human_name': human_name,
                    'value': feat_val,
                    'shap': val,
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
