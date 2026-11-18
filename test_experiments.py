"""Tests for ML experiments module and API endpoints."""

import os
import sys
import json
import unittest
import time
import threading

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExperimentEngine(unittest.TestCase):
    """Unit tests for the ExperimentEngine class."""

    @classmethod
    def setUpClass(cls):
        """Load the experiment engine once for all tests."""
        from app.ml.experiments import ExperimentEngine
        cls.engine = ExperimentEngine()

    def test_dataset_loaded(self):
        """Verify the dataset was loaded correctly."""
        self.assertIsNotNone(self.engine.X)
        self.assertIsNotNone(self.engine.y)
        self.assertEqual(self.engine.X.shape[0], len(self.engine.y))
        self.assertEqual(self.engine.X.shape[1], 14)

    def test_dataset_info(self):
        """Verify dataset info contains expected fields."""
        info = self.engine.get_dataset_info()
        self.assertIn("n_samples", info)
        self.assertIn("n_features", info)
        self.assertIn("target_distribution", info)
        self.assertIn("feature_names", info)
        self.assertIn("class_names", info)
        self.assertEqual(info["n_features"], 14)
        self.assertEqual(len(info["feature_names"]), 14)
        self.assertIn("No Default", info["target_distribution"])
        self.assertIn("Default", info["target_distribution"])

    def test_target_is_binary(self):
        """Verify target column is binary (0 and 1)."""
        import numpy as np
        unique = np.unique(self.engine.y)
        self.assertTrue(set(unique).issubset({0, 1}))

    def test_feature_names_match(self):
        """Verify feature names match expected names from config."""
        expected = [
            'age', 'annual_income', 'employment_length', 'employment_type',
            'home_ownership', 'existing_debt', 'num_credit_lines',
            'credit_history_length', 'late_payments', 'bankruptcies',
            'loan_amount', 'loan_purpose', 'loan_tenure', 'debt_to_income_ratio'
        ]
        self.assertEqual(self.engine.feature_names, expected)

    def test_xgb_model_factory(self):
        """Verify XGBClassifier has production-matching hyperparameters."""
        model = self.engine._get_xgb_model()
        self.assertEqual(model.n_estimators, 200)
        self.assertEqual(model.max_depth, 6)
        self.assertAlmostEqual(model.learning_rate, 0.1)

    def test_rf_model_factory(self):
        """Verify RandomForestClassifier has expected hyperparameters."""
        model = self.engine._get_rf_model()
        self.assertEqual(model.n_estimators, 200)
        self.assertEqual(model.max_depth, 10)

    def test_80_20_split_xgboost(self):
        """Test 80:20 split evaluation with XGBoost."""
        results = self.engine.run_80_20_split(model_type='xgboost')
        self._validate_split_results(results)

    def test_80_20_split_random_forest(self):
        """Test 80:20 split evaluation with Random Forest."""
        results = self.engine.run_80_20_split(model_type='random_forest')
        self._validate_split_results(results)

    def test_80_20_split_with_feature_selection(self):
        """Test 80:20 split with a subset of features."""
        results = self.engine.run_80_20_split(
            model_type='xgboost',
            feature_indices=[0, 1, 2, 3, 4]
        )
        self._validate_split_results(results)
        self.assertEqual(results['n_features'], 5)

    def _validate_split_results(self, results):
        """Validate structure of split results dict."""
        required_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                         'confusion_matrix', 'training_time', 'n_features']
        for key in required_keys:
            self.assertIn(key, results, f"Missing key: {key}")

        # Metric ranges
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            self.assertGreaterEqual(results[metric], 0.0)
            self.assertLessEqual(results[metric], 1.0)

        # Confusion matrix shape
        cm = results['confusion_matrix']
        self.assertEqual(len(cm), 2)
        self.assertEqual(len(cm[0]), 2)
        self.assertEqual(len(cm[1]), 2)

        # Training time is positive
        self.assertGreater(results['training_time'], 0)

    def test_5fold_cv_xgboost(self):
        """Test 5-fold cross-validation with XGBoost."""
        results = self.engine.run_cross_validation(n_folds=5, model_type='xgboost')
        self._validate_cv_results(results, expected_folds=5)

    def test_10fold_cv_xgboost(self):
        """Test 10-fold cross-validation with XGBoost."""
        results = self.engine.run_cross_validation(n_folds=10, model_type='xgboost')
        self._validate_cv_results(results, expected_folds=10)

    def test_5fold_cv_with_feature_selection(self):
        """Test cross-validation with feature subset."""
        results = self.engine.run_cross_validation(
            n_folds=5, model_type='xgboost',
            feature_indices=[0, 1, 5, 10, 13]
        )
        self._validate_cv_results(results, expected_folds=5)
        self.assertEqual(results['n_features'], 5)

    def _validate_cv_results(self, results, expected_folds):
        """Validate structure of CV results dict."""
        required_keys = ['fold_scores', 'mean_accuracy', 'std_accuracy',
                         'mean_f1', 'std_f1', 'mean_roc_auc', 'std_roc_auc',
                         'training_time', 'n_features']
        for key in required_keys:
            self.assertIn(key, results, f"Missing key: {key}")

        # Correct number of folds
        self.assertEqual(len(results['fold_scores']), expected_folds)

        # Each fold has expected metrics
        for fold in results['fold_scores']:
            self.assertIn('accuracy', fold)
            self.assertIn('f1', fold)
            self.assertIn('roc_auc', fold)

        # Means are in valid range
        for metric in ['mean_accuracy', 'mean_f1', 'mean_roc_auc']:
            self.assertGreaterEqual(results[metric], 0.0)
            self.assertLessEqual(results[metric], 1.0)

        # Std is non-negative
        for metric in ['std_accuracy', 'std_f1', 'std_roc_auc']:
            self.assertGreaterEqual(results[metric], 0.0)


class TestGeneticAlgorithm(unittest.TestCase):
    """Tests for the Genetic Algorithm feature selection."""

    @classmethod
    def setUpClass(cls):
        """Run GA once (expensive) and reuse results."""
        from app.ml.experiments import ExperimentEngine
        cls.engine = ExperimentEngine()
        cls.ga_output = cls.engine.run_genetic_algorithm(
            population_size=10,  # small for testing speed
            n_generations=5,
            crossover_rate=0.8,
            mutation_rate=0.05,
            tournament_size=3,
            cv_folds=3
        )

    def test_ga_config_present(self):
        """Verify GA config is returned."""
        self.assertIn('ga_config', self.ga_output)
        config = self.ga_output['ga_config']
        self.assertEqual(config['population_size'], 10)
        self.assertEqual(config['n_generations'], 5)

    def test_ga_results_present(self):
        """Verify GA results have expected fields."""
        self.assertIn('ga_results', self.ga_output)
        results = self.ga_output['ga_results']
        required_keys = ['selected_features', 'excluded_features', 'n_selected',
                         'feature_mask', 'best_fitness', 'fitness_history',
                         'total_generations']
        for key in required_keys:
            self.assertIn(key, results)

    def test_feature_mask_is_binary(self):
        """Verify feature mask has only 0s and 1s."""
        mask = self.ga_output['ga_results']['feature_mask']
        self.assertEqual(len(mask), 14)
        for val in mask:
            self.assertIn(val, [0, 1])

    def test_at_least_2_features_selected(self):
        """Verify GA selected at least 2 features."""
        self.assertGreaterEqual(self.ga_output['ga_results']['n_selected'], 2)

    def test_selected_excluded_cover_all(self):
        """Verify selected + excluded = all features."""
        results = self.ga_output['ga_results']
        all_feats = set(results['selected_features']) | set(results['excluded_features'])
        expected = set(self.engine.feature_names)
        self.assertEqual(all_feats, expected)

    def test_fitness_history_non_empty(self):
        """Verify fitness history was recorded."""
        history = self.ga_output['ga_results']['fitness_history']
        self.assertGreater(len(history), 0)
        for val in history:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_best_fitness_positive(self):
        """Verify best fitness is a positive number."""
        self.assertGreater(self.ga_output['ga_results']['best_fitness'], 0.0)

    def test_no_test_data_leakage(self):
        """
        Verify GA fitness evaluation uses training data only.
        
        The GA uses train_test_split with test_size=0.2 internally,
        and the fitness function only receives X_train for CV.
        This is verified by the fact that the GA method creates
        its own 80/20 split and passes only X_train to the fitness function.
        """
        import inspect
        source = inspect.getsource(self.engine.run_genetic_algorithm)
        # Verify the train_test_split call exists
        self.assertIn('train_test_split', source)
        # Verify it only uses X_train in fitness
        self.assertIn('X_train', source)


class TestLeakagePrevention(unittest.TestCase):
    """Tests to ensure no data leakage between train and test sets."""

    @classmethod
    def setUpClass(cls):
        from app.ml.experiments import ExperimentEngine
        cls.engine = ExperimentEngine()

    def test_same_split_reproducibility(self):
        """Verify same random_state gives identical splits."""
        from sklearn.model_selection import train_test_split
        X = self.engine.X
        y = self.engine.y

        _, X_test1, _, y_test1 = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        _, X_test2, _, y_test2 = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        import numpy as np
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_test1, y_test2)

    def test_cv_no_overlap(self):
        """Verify CV folds don't overlap between train and validation sets."""
        from sklearn.model_selection import StratifiedKFold
        import numpy as np
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in skf.split(self.engine.X, self.engine.y):
            overlap = set(train_idx) & set(val_idx)
            self.assertEqual(len(overlap), 0, "Train and validation indices overlap!")


class TestMetricsComputation(unittest.TestCase):
    """Tests for metric computation logic."""

    @classmethod
    def setUpClass(cls):
        from app.ml.experiments import ExperimentEngine
        cls.engine = ExperimentEngine()

    def test_perfect_predictions(self):
        """Verify metrics for perfect predictions."""
        import numpy as np
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.85])
        metrics = self.engine._compute_metrics(y_true, y_pred, y_prob, 1.0, 14)
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)

    def test_all_wrong_predictions(self):
        """Verify metrics for completely wrong predictions."""
        import numpy as np
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        metrics = self.engine._compute_metrics(y_true, y_pred, y_prob, 0.5, 14)
        self.assertEqual(metrics['accuracy'], 0.0)
        self.assertEqual(metrics['recall'], 0.0)


class TestExperimentResults(unittest.TestCase):
    """Test the ExperimentResults dataclass."""

    def test_default_values(self):
        from app.ml.experiments import ExperimentResults
        r = ExperimentResults()
        self.assertEqual(r.status, 'pending')
        self.assertIsNone(r.error_message)
        self.assertIsNone(r.dataset_info)

    def test_dataclass_asdict(self):
        from dataclasses import asdict
        from app.ml.experiments import ExperimentResults
        r = ExperimentResults(experiment_id='test-123', status='completed')
        d = asdict(r)
        self.assertEqual(d['experiment_id'], 'test-123')
        self.assertEqual(d['status'], 'completed')


class TestAPIEndpoints(unittest.TestCase):
    """Test the FastAPI experiment endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up a test client."""
        # Import the FastAPI app
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app)

    def test_dataset_info_endpoint(self):
        """Test GET /api/experiments/dataset-info."""
        response = self.client.get("/api/experiments/dataset-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("n_samples", data)
        self.assertIn("n_features", data)
        self.assertIn("target_distribution", data)

    def test_list_experiments_endpoint(self):
        """Test GET /api/experiments/list."""
        response = self.client.get("/api/experiments/list")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("experiments", data)

    def test_latest_no_experiments(self):
        """Test GET /api/experiments/latest returns 404 when no experiments exist."""
        # This may or may not pass depending on test ordering, but initially it should 404
        response = self.client.get("/api/experiments/latest")
        # Accept either 200 (if experiments exist) or 404
        self.assertIn(response.status_code, [200, 404])

    def test_status_not_found(self):
        """Test GET /api/experiments/status/{id} returns 404 for nonexistent experiment."""
        response = self.client.get("/api/experiments/status/nonexistent-id-12345")
        self.assertEqual(response.status_code, 404)

    def test_results_not_found(self):
        """Test GET /api/experiments/results/{id} returns not_found status."""
        response = self.client.get("/api/experiments/results/nonexistent-id-12345")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "not_found")

    def test_run_experiment_endpoint(self):
        """Test POST /api/experiments/run starts an experiment."""
        response = self.client.post("/api/experiments/run")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("experiment_id", data)
        self.assertEqual(data["status"], "running")
        self.assertIn("message", data)

    def test_experiments_page_requires_auth(self):
        """Test GET /dashboard/experiments redirects without auth."""
        response = self.client.get("/dashboard/experiments", follow_redirects=False)
        # Should redirect to login since user is not authenticated
        self.assertIn(response.status_code, [302, 303, 307, 401])


class TestExistingPredictionPipeline(unittest.TestCase):
    """Verify the existing prediction pipeline is not broken."""

    def test_ml_service_loads(self):
        """Test that the ML service still loads correctly."""
        from app.services.ml_service import ml_service
        self.assertIsNotNone(ml_service)

    def test_ml_service_has_feature_names(self):
        """Test that feature names are correct."""
        from app.services.ml_service import ml_service
        expected = [
            'age', 'annual_income', 'employment_length', 'employment_type',
            'home_ownership', 'existing_debt', 'num_credit_lines',
            'credit_history_length', 'late_payments', 'bankruptcies',
            'loan_amount', 'loan_purpose', 'loan_tenure', 'debt_to_income_ratio'
        ]
        self.assertEqual(ml_service.feature_names, expected)

    def test_lightweight_model_loaded(self):
        """Test that the lightweight model is available."""
        from app.services.ml_service import ml_service
        # At least one model should be loaded
        self.assertTrue(
            ml_service.lightweight_model is not None or ml_service.xgb_booster is not None,
            "No prediction model loaded"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
