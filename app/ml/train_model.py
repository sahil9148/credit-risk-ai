import os
import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import shap

FEATURE_NAMES = [
    'age', 'annual_income', 'employment_length', 'employment_type',
    'home_ownership', 'existing_debt', 'num_credit_lines',
    'credit_history_length', 'late_payments', 'bankruptcies',
    'loan_amount', 'loan_purpose', 'loan_tenure', 'debt_to_income_ratio'
]

def train_model():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credit_data.csv')
    
    if not os.path.exists(data_path):
        print("Data not found, generating...")
        # Import generate_data in a way that works both as script and module
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        from generate_data import generate_dataset
        generate_dataset()

    df = pd.read_csv(data_path)
    
    X = df[FEATURE_NAMES]
    y = df['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42, 
        eval_metric='logloss'
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    print("\nCreating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
    explainer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'explainer.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(explainer_path, 'wb') as f:
        pickle.dump(explainer, f)
        
    print(f"\nModel saved to {model_path}")
    print(f"Explainer saved to {explainer_path}")
    print("\nTraining complete!")

if __name__ == '__main__':
    train_model()
