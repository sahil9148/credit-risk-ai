# 🏦 Credit Risk AI — Explainable Credit Scoring Platform

An automated credit risk assessment platform powered by **FastAPI**, **XGBoost**, and **SHAP (SHapley Additive exPlanations)** with an interactive multi-step wizard, real-time risk explanations, counterfactual "What-If" analysis, and personalized loan product matching.

---

## 🌟 Key Features

- **XGBoost Classifier**: Predicts default probability with ~91.5% accuracy and 0.954 AUC-ROC.
- **SHAP TreeExplainer**: Provides mathematical, transparent feature-level waterfall charts (game-theoretic Shapley values).
- **Interactive What-If Simulation**: Adjust financial metrics in real-time to simulate score improvements.
- **Personalized Loan Matching**: Matches eligible loan products with tenure-based EMI calculations.
- **Secure Authentication**: JWT-based authentication stored in `httpOnly` cookies with `bcrypt` password hashing.
- **Admin Dashboard**: Portfolio analytics, approval rate tracking, and loan status review.

---

## 🚀 One-Click Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

1. Push this repository to your GitHub account.
2. Go to [Vercel Dashboard](https://vercel.com) and click **"Add New Project"**.
3. Select your GitHub repository (`credit-risk-ai`).
4. Keep the default settings and click **Deploy**.
5. Vercel automatically detects `vercel.json` and `api/index.py` using Python serverless.

> **Note on Database**:
> - By default, the application runs on the included SQLite database (copies to `/tmp` on serverless).
> - For multi-instance persistent production storage, set the `DATABASE_URL` environment variable in Vercel to a hosted PostgreSQL instance (e.g., Supabase, Neon, or Railway).

---

## 💻 Local Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/sahil9148/credit-risk-ai.git
cd credit-risk-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the development server
```bash
python launch_server.py
```
Or with Uvicorn directly:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

- **Default Admin Account**: `admin@creditrisk.ai` / `admin123`
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📁 Project Structure

```
credit-risk-ai/
├── api/
│   └── index.py             # Vercel serverless entrypoint
├── app/
│   ├── main.py              # FastAPI application initialization & routing
│   ├── config.py            # Environment & database configuration
│   ├── database.py          # SQLAlchemy engine & session factory
│   ├── ml/                  # ML training script, data generator & pickle models
│   ├── models/              # User, LoanApplication, RiskAssessment ORM models
│   ├── routers/             # Auth, applications, assessments, admin & page routers
│   ├── schemas/             # Pydantic validation schemas
│   ├── services/            # Auth, ML prediction & loan recommendation services
│   ├── static/              # CSS & JS assets
│   └── templates/           # Jinja2 HTML templates
├── credit_risk.db           # SQLite database
├── vercel.json              # Vercel deployment routing configuration
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 📄 License
MIT License. Created for academic demonstration and portfolio showcase.
