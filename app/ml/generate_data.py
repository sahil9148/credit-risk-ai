import numpy as np
import pandas as pd
import os

def generate_dataset():
    np.random.seed(42)
    n_samples = 10000

    age = np.clip(np.random.normal(40, 12, n_samples), 18, 80).astype(int)
    annual_income = np.clip(np.random.lognormal(mean=10.9, sigma=0.5, size=n_samples), 15000, 500000)
    employment_length = np.random.randint(0, 41, n_samples)
    employment_type = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.05, 0.05, 0.1, 0.2, 0.6])
    home_ownership = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.05, 0.35, 0.4, 0.2])
    existing_debt = np.clip(np.random.normal(30000, 40000, n_samples), 0, 200000)
    num_credit_lines = np.random.randint(0, 31, n_samples)
    credit_history_length = np.random.randint(0, 41, n_samples)
    late_payments = np.random.poisson(lam=0.5, size=n_samples)
    late_payments = np.clip(late_payments, 0, 20)
    bankruptcies = np.random.poisson(lam=0.1, size=n_samples)
    bankruptcies = np.clip(bankruptcies, 0, 3)
    loan_amount = np.clip(np.random.normal(50000, 50000, n_samples), 1000, 500000)
    loan_purpose = np.random.randint(0, 7, n_samples)
    loan_tenure = np.random.choice([12, 24, 36, 48, 60, 72, 120, 240, 360], size=n_samples)
    debt_to_income_ratio = existing_debt / np.maximum(annual_income, 1)

    df = pd.DataFrame({
        'age': age,
        'annual_income': annual_income,
        'employment_length': employment_length,
        'employment_type': employment_type,
        'home_ownership': home_ownership,
        'existing_debt': existing_debt,
        'num_credit_lines': num_credit_lines,
        'credit_history_length': credit_history_length,
        'late_payments': late_payments,
        'bankruptcies': bankruptcies,
        'loan_amount': loan_amount,
        'loan_purpose': loan_purpose,
        'loan_tenure': loan_tenure,
        'debt_to_income_ratio': debt_to_income_ratio
    })

    # Calculate risk score
    risk_score = (
        (df['debt_to_income_ratio'] - 0.3) * 5 +
        df['late_payments'] * 1.5 +
        df['bankruptcies'] * 2.5 +
        (df['employment_type'] == 0) * 1.5 -
        (np.log1p(df['annual_income']) - 10) * 0.5 -
        (df['credit_history_length'] - 5) * 0.1 +
        (df['loan_amount'] / np.maximum(df['annual_income'], 1)) * 2 -
        2.5  # Base intercept
    )

    # Sigmoid to probability
    prob_default = 1 / (1 + np.exp(-risk_score))
    
    # Generate labels
    df['default'] = np.random.binomial(1, prob_default)

    output_path = os.path.join(os.path.dirname(__file__), 'credit_data.csv')
    df.to_csv(output_path, index=False)
    
    print("Generated data statistics:")
    print(df.describe())
    print(f"\nTarget distribution:\n{df['default'].value_counts(normalize=True)}")
    print(f"\nData saved to {output_path}")

if __name__ == '__main__':
    generate_dataset()
