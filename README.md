# 🏦 Credit Risk AI

AI-powered credit risk assessment platform with XGBoost, SHAP explainability, genetic algorithm feature selection, and comprehensive ML experimentation dashboard.

## Features

- **AI Credit Scoring** — XGBoost-based credit risk prediction with probability of default
- **SHAP Explainability** — Feature contribution analysis showing why a score was given
- **Loan Recommendations** — Personalized loan product suggestions based on risk profile
- **ML Experiments Dashboard** — Comprehensive model evaluation with cross-validation, genetic algorithm feature selection, and multi-model comparison
- **Admin Panel** — Manage users and view all applications
- **Responsive UI** — Tailwind CSS + Alpine.js dashboard

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Python 3.11 |
| ML | XGBoost, scikit-learn, SHAP |
| Feature Selection | Genetic Algorithm (custom implementation) |
| Database | SQLAlchemy + SQLite |
| Frontend | Jinja2, Tailwind CSS, Alpine.js, Chart.js |
| Auth | JWT + bcrypt |

## Quick Start

```bash
# Clone
git clone https://github.com/sahil9148/credit-risk-ai.git
cd credit-risk-ai

# Install dependencies
pip install -r requirements.txt

# Train the model (if not already trained)
python -m app.ml.train_model

# Start the server
python launch_server.py
```

Open **http://localhost:8000** in your browser.

### Default Admin Credentials
- Email: `admin@creditrisk.ai`
- Password: `admin123`

## ML Experiments

The platform includes a dedicated ML Experiments Dashboard (`/dashboard/experiments`) that evaluates models across multiple strategies:

### Experiments Performed

1. **80:20 Train-Test Split** — Stratified split evaluation with accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
2. **5-Fold Cross-Validation** — StratifiedKFold with per-fold scores and aggregated statistics
3. **10-Fold Cross-Validation** — Same methodology with 10 folds for more robust estimates
4. **Genetic Algorithm Feature Selection** — Binary chromosome GA optimizing F1-score to select the best feature subset
5. **GA + XGBoost** — XGBoost trained on GA-selected features
6. **GA + Random Forest** — RandomForestClassifier trained on GA-selected features for comparison

### Running Experiments

**Via the Dashboard:**
1. Log in and navigate to **ML Experiments** in the sidebar
2. Click **Run New Experiment**
3. Wait for completion (the GA takes ~2-5 minutes)
4. View results, charts, and export CSV/JSON

**Via the API:**
```bash
# Start an experiment
curl -X POST http://localhost:8000/api/experiments/run

# Check status
curl http://localhost:8000/api/experiments/status/{experiment_id}

# Get results
curl http://localhost:8000/api/experiments/results/{experiment_id}

# Export as JSON
curl http://localhost:8000/api/experiments/results/{experiment_id}/export/json

# Export comparison as CSV
curl http://localhost:8000/api/experiments/results/{experiment_id}/export/csv
```

**Via Python:**
```python
from app.ml.experiments import experiment_engine

# Run all experiments
results = experiment_engine.run_all_experiments()

# View comparison table
for row in results.comparison_table:
    print(f"{row['Model']}: F1={row['Split_F1']:.4f}, AUC={row['Split_ROC_AUC']:.4f}")
```

### Genetic Algorithm Details

| Parameter | Value |
|-----------|-------|
| Population Size | 30 |
| Max Generations | 50 |
| Crossover Rate | 0.8 |
| Mutation Rate | 0.05 |
| Tournament Size | 3 |
| Fitness Metric | Mean CV F1-Score |
| Early Stopping | 10 generations without improvement |

- **Chromosome**: Binary array of length 14 (one gene per feature)
- **Selection**: Tournament selection
- **Crossover**: Single-point crossover
- **Mutation**: Bit-flip mutation
- **Elitism**: Best individual preserved each generation
- **Leakage Prevention**: Fitness evaluated only on 80% training split using inner CV

### Reproducibility

All experiments use `random_state=42` for:
- Train-test split
- Cross-validation fold generation
- XGBoost model training
- Random Forest model training
- GA population initialization

## Running Tests

```bash
python -m unittest test_experiments -v
```

Tests cover:
- Dataset loading and validation
- Model factory hyperparameters
- 80:20 split evaluation
- 5-fold and 10-fold cross-validation
- Genetic algorithm correctness
- Data leakage prevention
- Metric computation
- API endpoint behavior
- Existing prediction pipeline preservation

## Project Structure

```
credit-risk-ai/
├── app/
│   ├── main.py                  # FastAPI application
│   ├── config.py                # Configuration
│   ├── database.py              # Database setup
│   ├── ml/
│   │   ├── credit_data.csv      # Dataset (10,000 samples)
│   │   ├── train_model.py       # Production model training
│   │   ├── experiments.py       # ML experiments engine
│   │   ├── model.pkl            # Trained XGBoost model
│   │   ├── model.json           # XGBoost model (JSON format)
│   │   └── explainer.pkl        # SHAP TreeExplainer
│   ├── models/                  # SQLAlchemy ORM models
│   ├── routers/
│   │   ├── pages.py             # Page routes
│   │   ├── experiments.py       # ML experiment API
│   │   ├── auth.py              # Authentication
│   │   ├── applications.py      # Loan applications
│   │   └── assessments.py       # Risk assessments
│   ├── services/
│   │   ├── ml_service.py        # Production prediction service
│   │   ├── auth_service.py      # Auth utilities
│   │   └── recommendation.py    # Loan recommendations
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard/
│   │   │   ├── experiments.html # ML Experiments dashboard
│   │   │   ├── index.html
│   │   │   ├── apply.html
│   │   │   ├── report.html
│   │   │   └── history.html
│   │   └── auth/
│   └── static/
├── test_experiments.py          # ML experiment tests
├── requirements.txt
├── launch_server.py
└── README.md
```

## Dataset

- **Samples**: 10,000 synthetic credit applications
- **Features**: 14 (age, annual_income, employment_length, employment_type, home_ownership, existing_debt, num_credit_lines, credit_history_length, late_payments, bankruptcies, loan_amount, loan_purpose, loan_tenure, debt_to_income_ratio)
- **Target**: `default` (binary: 0 = No Default, 1 = Default)
- **Positive Class**: Default (class 1)

## License

MIT
