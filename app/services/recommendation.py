from __future__ import annotations
from typing import List, Dict

def get_loan_recommendations(credit_score: int, loan_amount: float, annual_income: float, loan_purpose: str) -> list[dict]:
    products = [
        {"name": "Premium Personal Loan", "min_score": 750, "rate_range": (8.5, 11.0), "max_amount": 2000000},
        {"name": "Standard Personal Loan", "min_score": 600, "rate_range": (12.0, 16.0), "max_amount": 1000000},
        {"name": "Secured Home Loan", "min_score": 650, "rate_range": (7.5, 9.5), "max_amount": 10000000, "purpose": "home"},
        {"name": "Auto Loan", "min_score": 600, "rate_range": (9.0, 13.0), "max_amount": 3000000, "purpose": "car"},
        {"name": "Education Loan", "min_score": 500, "rate_range": (10.0, 14.0), "max_amount": 5000000, "purpose": "education"},
        {"name": "Business Loan", "min_score": 700, "rate_range": (11.0, 15.0), "max_amount": 5000000, "purpose": "business"},
        {"name": "Debt Consolidation Loan", "min_score": 550, "rate_range": (13.0, 18.0), "max_amount": 1500000, "purpose": "debt_consolidation"},
        {"name": "Emergency Personal Loan", "min_score": 400, "rate_range": (18.0, 24.0), "max_amount": 500000},
    ]

    eligible = []
    for p in products:
        if credit_score >= p['min_score'] and loan_amount <= p['max_amount']:
            eligible.append(p)

    results = []
    for p in eligible:
        eligible_amount = min(loan_amount, p['max_amount'])
        
        max_score = 900
        score_ratio = min(1.0, max(0.0, (credit_score - p['min_score']) / (max_score - p['min_score'])))
        min_rate, max_rate = p['rate_range']
        estimated_rate = max_rate - (score_ratio * (max_rate - min_rate))
        
        emi_options = {}
        monthly_rate = estimated_rate / (12 * 100)
        for tenure in [12, 24, 36, 60]:
            if monthly_rate > 0:
                emi = (eligible_amount * monthly_rate * ((1 + monthly_rate)**tenure)) / (((1 + monthly_rate)**tenure) - 1)
            else:
                emi = eligible_amount / tenure
            emi_options[str(tenure)] = round(emi, 2)
            
        match_score = 50
        if p.get('purpose') == loan_purpose.lower():
            match_score += 30
        
        match_score += max(0, 20 - (estimated_rate - 7))
        match_score = min(100, max(0, match_score))
        
        results.append({
            "name": p['name'],
            "description": f"Loan up to {p['max_amount']} at {p['rate_range'][0]}-{p['rate_range'][1]}%",
            "min_score": p['min_score'],
            "rate_range": f"{p['rate_range'][0]}-{p['rate_range'][1]}%",
            "max_amount": p['max_amount'],
            "eligible_amount": eligible_amount,
            "estimated_rate": round(estimated_rate, 2),
            "emi_options": emi_options,
            "match_score": round(match_score, 1),
            "is_recommended": False
        })
        
    results.sort(key=lambda x: x['match_score'], reverse=True)
    if len(results) > 0:
        results[0]['is_recommended'] = True
    if len(results) > 1:
        results[1]['is_recommended'] = True
        
    return results

def get_improvement_tips(credit_score: int, risk_factors: list[str]) -> list[str]:
    tips = []
    
    rf_lower = " ".join(risk_factors).lower()
    
    if "debt" in rf_lower or "ratio" in rf_lower:
        tips.append("Reduce your debt-to-income ratio below 40% to improve your score")
    if "history" in rf_lower or "length" in rf_lower:
        tips.append("Build a longer credit history by keeping old accounts open")
    if "late" in rf_lower or "payment" in rf_lower:
        tips.append("Make timely payments to reduce your late payment count")
    if "bankruptc" in rf_lower:
        tips.append("Time will heal bankruptcy records; focus on building clean new credit")
    if "lines" in rf_lower:
        tips.append("Avoid opening too many new credit lines at once")
        
    if not tips:
        if credit_score < 600:
            tips.append("Focus on paying down existing balances to improve your profile")
        elif credit_score < 750:
            tips.append("Continue your good payment history to reach excellent tier")
        else:
            tips.append("Maintain your excellent financial habits")
            
    return tips
