import os

# Hardcoded for demo
SECRET_KEY = "a4f8d9c2e1b0a5f7d8c9e0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database configuration: support external DB (Postgres/Supabase), Vercel serverless /tmp, or local SQLite
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    if os.getenv("VERCEL"):
        import shutil
        tmp_db = "/tmp/credit_risk.db"
        src_db = os.path.join(BASE_DIR, "credit_risk.db")
        if os.path.exists(src_db) and not os.path.exists(tmp_db):
            try:
                shutil.copyfile(src_db, tmp_db)
            except Exception:
                pass
        DATABASE_URL = f"sqlite:///{tmp_db}"
    else:
        DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'credit_risk.db')}"

# Handle Postgres protocol prefix for SQLAlchemy if needed
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "model.pkl")
EXPLAINER_PATH = os.path.join(os.path.dirname(__file__), "ml", "explainer.pkl")

FEATURE_NAMES = [
    'age', 
    'annual_income', 
    'employment_length', 
    'employment_type', 
    'home_ownership', 
    'existing_debt', 
    'num_credit_lines', 
    'credit_history_length', 
    'late_payments', 
    'bankruptcies', 
    'loan_amount', 
    'loan_purpose', 
    'loan_tenure', 
    'debt_to_income_ratio'
]
