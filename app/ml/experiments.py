"""
Core ML experiments module for the Credit Risk AI project.

Provides a comprehensive experiment engine that runs XGBoost and Random Forest
models with genetic algorithm-based feature selection, evaluating performance
across 80:20 splits and cross-validation strategies.
"""

import csv
import io
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


FEATURE_NAMES = [
    "age",
    "annual_income",
    "employment_length",
    "employment_type",
    "home_ownership",
    "existing_debt",
    "num_credit_lines",
    "credit_history_length",
    "late_payments",
    "bankruptcies",
    "loan_amount",
    "loan_purpose",
    "loan_tenure",
    "debt_to_income_ratio",
]


@dataclass
class ExperimentResults:
    """Structured container for all experiment results."""

    experiment_id: str = ""
    experiment_name: str = "Credit Risk ML Experiment"
    status: str = "pending"  # pending | running | completed | error
    error_message: Optional[str] = None

    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Dataset metadata
    dataset_info: Optional[Dict] = None

    # XGBoost with all features
    split_results: Optional[Dict] = None
    cv5_results: Optional[Dict] = None
    cv10_results: Optional[Dict] = None

    # Genetic algorithm configuration and results
    ga_config: Optional[Dict] = None
    ga_results: Optional[Dict] = None

    # XGBoost with GA-selected features
    ga_xgboost_split_results: Optional[Dict] = None
    ga_xgboost_cv5_results: Optional[Dict] = None
    ga_xgboost_cv10_results: Optional[Dict] = None

    # Random Forest with GA-selected features
    ga_rf_split_results: Optional[Dict] = None
    ga_rf_cv5_results: Optional[Dict] = None
    ga_rf_cv10_results: Optional[Dict] = None

    # Fair comparison table across all model/feature combos
    comparison_table: Optional[List[Dict]] = None


class ExperimentEngine:
    """Engine that orchestrates all ML experiments for credit risk modelling.

    Loads the dataset once and exposes methods to run individual evaluation
    strategies (train/test split, k-fold CV) as well as genetic-algorithm-based
    feature selection.  The ``run_all_experiments`` method executes the full
    pipeline and returns a populated ``ExperimentResults`` instance.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        """Load the credit-risk dataset and initialise the results store."""
        data_path = os.path.join(os.path.dirname(__file__), "credit_data.csv")
        self.df = pd.read_csv(data_path)
        self.feature_names: List[str] = FEATURE_NAMES
        self.target_column: str = "default"

        self.X: np.ndarray = self.df[self.feature_names].values
        self.y: np.ndarray = self.df[self.target_column].values

        # Experiment results store: experiment_id -> ExperimentResults
        self.results: Dict[str, ExperimentResults] = {}

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------
    def get_dataset_info(self) -> dict:
        """Return a summary of the loaded dataset.

        Returns
        -------
        dict
            Keys: n_samples, n_features, target_distribution, feature_names,
            class_names.
        """
        n_samples = len(self.y)
        counts = {int(k): int(v) for k, v in zip(*np.unique(self.y, return_counts=True))}
        distribution = {}
        for cls_val, cls_label in [(0, "No Default"), (1, "Default")]:
            count = counts.get(cls_val, 0)
            distribution[cls_label] = {
                "count": count,
                "percentage": round(count / n_samples * 100, 2) if n_samples else 0.0,
            }

        return {
            "n_samples": n_samples,
            "n_features": len(self.feature_names),
            "target_distribution": distribution,
            "feature_names": list(self.feature_names),
            "class_names": ["No Default", "Default"],
        }

    # ------------------------------------------------------------------
    # Model factories
    # ------------------------------------------------------------------
    def _get_xgb_model(self) -> XGBClassifier:
        """Return a new XGBClassifier matching the production hyperparameters."""
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )

    def _get_rf_model(self) -> RandomForestClassifier:
        """Return a new RandomForestClassifier with default experiment params."""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        training_time: float,
        n_features: int,
    ) -> dict:
        """Compute classification metrics for a single evaluation.

        Parameters
        ----------
        y_true : array-like
            Ground-truth labels.
        y_pred : array-like
            Predicted labels.
        y_prob : array-like
            Predicted probabilities for the positive class.
        training_time : float
            Wall-clock seconds spent training.
        n_features : int
            Number of features used.

        Returns
        -------
        dict
            accuracy, precision, recall, f1, roc_auc, confusion_matrix,
            training_time, n_features.
        """
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "accuracy": float(round(accuracy_score(y_true, y_pred), 6)),
            "precision": float(round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 6)),
            "recall": float(round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 6)),
            "f1": float(round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 6)),
            "roc_auc": float(round(roc_auc_score(y_true, y_prob), 6)),
            "confusion_matrix": cm,
            "training_time": float(round(training_time, 4)),
            "n_features": n_features,
        }

    # ------------------------------------------------------------------
    # Individual experiment runners
    # ------------------------------------------------------------------
    def run_80_20_split(
        self,
        model_type: str = "xgboost",
        feature_indices: Optional[List[int]] = None,
    ) -> dict:
        """Train and evaluate on an 80/20 stratified split.

        Parameters
        ----------
        model_type : str
            ``'xgboost'`` or ``'random_forest'``.
        feature_indices : list[int] or None
            Column indices to select.  ``None`` means all features.

        Returns
        -------
        dict
            Metrics dictionary produced by ``_compute_metrics``.
        """
        X = self.X[:, feature_indices] if feature_indices is not None else self.X
        n_features = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        model = self._get_xgb_model() if model_type == "xgboost" else self._get_rf_model()

        start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        return self._compute_metrics(y_test, y_pred, y_prob, training_time, n_features)

    def run_cross_validation(
        self,
        n_folds: int,
        model_type: str = "xgboost",
        feature_indices: Optional[List[int]] = None,
    ) -> dict:
        """Run stratified k-fold cross-validation.

        Parameters
        ----------
        n_folds : int
            Number of CV folds (e.g. 5 or 10).
        model_type : str
            ``'xgboost'`` or ``'random_forest'``.
        feature_indices : list[int] or None
            Column indices to select.

        Returns
        -------
        dict
            fold_scores, mean/std for accuracy, f1, roc_auc, training_time,
            n_features.
        """
        X = self.X[:, feature_indices] if feature_indices is not None else self.X
        n_features = X.shape[1]

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_accuracies: List[float] = []
        fold_f1s: List[float] = []
        fold_roc_aucs: List[float] = []
        fold_scores: List[Dict] = []
        total_training_time = 0.0

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, self.y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            model = self._get_xgb_model() if model_type == "xgboost" else self._get_rf_model()

            start = time.time()
            model.fit(X_train, y_train)
            fold_time = time.time() - start
            total_training_time += fold_time

            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            acc = float(accuracy_score(y_val, y_pred))
            f1 = float(f1_score(y_val, y_pred, pos_label=1, zero_division=0))
            auc = float(roc_auc_score(y_val, y_prob))

            fold_accuracies.append(acc)
            fold_f1s.append(f1)
            fold_roc_aucs.append(auc)
            fold_scores.append(
                {
                    "fold": fold_idx + 1,
                    "accuracy": round(acc, 6),
                    "f1": round(f1, 6),
                    "roc_auc": round(auc, 6),
                    "training_time": round(fold_time, 4),
                }
            )

        return {
            "fold_scores": fold_scores,
            "mean_accuracy": float(round(np.mean(fold_accuracies), 6)),
            "std_accuracy": float(round(np.std(fold_accuracies), 6)),
            "mean_f1": float(round(np.mean(fold_f1s), 6)),
            "std_f1": float(round(np.std(fold_f1s), 6)),
            "mean_roc_auc": float(round(np.mean(fold_roc_aucs), 6)),
            "std_roc_auc": float(round(np.std(fold_roc_aucs), 6)),
            "training_time": float(round(total_training_time, 4)),
            "n_features": n_features,
        }

    # ------------------------------------------------------------------
    # Genetic Algorithm for feature selection
    # ------------------------------------------------------------------
    def run_genetic_algorithm(
        self,
        population_size: int = 30,
        n_generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        tournament_size: int = 3,
        cv_folds: int = 5,
    ) -> dict:
        """Run a genetic algorithm to select the best feature subset.

        The GA optimises the mean cross-validated F1-score of XGBoost on the
        80 % training split.  It never touches the 20 % test set.

        Parameters
        ----------
        population_size : int
            Number of individuals per generation.
        n_generations : int
            Maximum generations to evolve.
        crossover_rate : float
            Probability of applying single-point crossover.
        mutation_rate : float
            Per-gene probability of bit-flip mutation.
        tournament_size : int
            Number of candidates in tournament selection.
        cv_folds : int
            Folds for the inner CV used to evaluate fitness.

        Returns
        -------
        dict
            ``ga_config`` and ``ga_results`` keys with full metadata.
        """
        np.random.seed(42)

        n_features = len(self.feature_names)

        # Use only the 80 % training split for fitness evaluation
        X_train, _, y_train, _ = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        # --- Fitness function ------------------------------------------------
        def _fitness(chromosome: np.ndarray) -> float:
            selected = np.where(chromosome == 1)[0]
            if len(selected) < 2:
                return 0.0
            X_sel = X_train[:, selected]
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            fold_f1s: List[float] = []
            for tr_idx, val_idx in skf.split(X_sel, y_train):
                model = self._get_xgb_model()
                model.fit(X_sel[tr_idx], y_train[tr_idx])
                preds = model.predict(X_sel[val_idx])
                fold_f1s.append(float(f1_score(y_train[val_idx], preds, pos_label=1, zero_division=0)))
            return float(np.mean(fold_f1s))

        # --- Population initialisation ---------------------------------------
        population: List[np.ndarray] = []
        for _ in range(population_size):
            chrom = np.random.randint(0, 2, size=n_features)
            # Ensure at least 2 features are selected
            while chrom.sum() < 2:
                chrom[np.random.randint(0, n_features)] = 1
            population.append(chrom)

        # Evaluate initial fitness
        fitness_values = [_fitness(ind) for ind in population]

        best_overall_fitness = max(fitness_values)
        best_overall_individual = population[np.argmax(fitness_values)].copy()
        fitness_history: List[float] = [best_overall_fitness]

        no_improvement_count = 0
        total_generations = 1  # already evaluated generation 0

        # --- Evolution loop ---------------------------------------------------
        for gen in range(1, n_generations):
            new_population: List[np.ndarray] = []

            # Elitism: carry over the best individual
            elite_idx = int(np.argmax(fitness_values))
            new_population.append(population[elite_idx].copy())

            # Fill the rest of the population
            while len(new_population) < population_size:
                # Tournament selection for two parents
                def _tournament() -> np.ndarray:
                    candidates = np.random.choice(population_size, size=tournament_size, replace=False)
                    best_c = candidates[0]
                    for c in candidates[1:]:
                        if fitness_values[c] > fitness_values[best_c]:
                            best_c = c
                    return population[best_c].copy()

                parent1 = _tournament()
                parent2 = _tournament()

                # Single-point crossover
                if np.random.rand() < crossover_rate:
                    point = np.random.randint(1, n_features)
                    child1 = np.concatenate([parent1[:point], parent2[point:]])
                    child2 = np.concatenate([parent2[:point], parent1[point:]])
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Bit-flip mutation
                for child in [child1, child2]:
                    for gene_idx in range(n_features):
                        if np.random.rand() < mutation_rate:
                            child[gene_idx] = 1 - child[gene_idx]
                    # Ensure at least 2 features selected after mutation
                    while child.sum() < 2:
                        child[np.random.randint(0, n_features)] = 1

                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)

            population = new_population[:population_size]
            fitness_values = [_fitness(ind) for ind in population]

            gen_best = max(fitness_values)
            fitness_history.append(gen_best)
            total_generations += 1

            if gen_best > best_overall_fitness:
                best_overall_fitness = gen_best
                best_overall_individual = population[np.argmax(fitness_values)].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= 10:
                break

        # --- Compile results --------------------------------------------------
        feature_mask = best_overall_individual.tolist()
        selected_indices = [i for i, v in enumerate(feature_mask) if v == 1]
        selected_features = [self.feature_names[i] for i in selected_indices]
        excluded_features = [self.feature_names[i] for i, v in enumerate(feature_mask) if v == 0]

        ga_config = {
            "population_size": population_size,
            "n_generations": n_generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "tournament_size": tournament_size,
            "cv_folds": cv_folds,
            "random_state": 42,
        }

        ga_results = {
            "selected_features": selected_features,
            "excluded_features": excluded_features,
            "n_selected": len(selected_features),
            "feature_mask": feature_mask,
            "best_fitness": float(round(best_overall_fitness, 6)),
            "fitness_history": [float(round(f, 6)) for f in fitness_history],
            "total_generations": total_generations,
        }

        return {"ga_config": ga_config, "ga_results": ga_results}

    # ------------------------------------------------------------------
    # Full experiment pipeline
    # ------------------------------------------------------------------
    def run_all_experiments(self, experiment_id: Optional[str] = None) -> ExperimentResults:
        """Execute the complete experiment pipeline.

        Steps
        -----
        1. Baseline XGBoost with all 14 features (split, 5-fold, 10-fold).
        2. Genetic Algorithm feature selection.
        3. XGBoost with GA-selected features (split, 5-fold, 10-fold).
        4. Random Forest with GA-selected features (split, 5-fold, 10-fold).
        5. Build a fair comparison table across all configurations.

        Parameters
        ----------
        experiment_id : str or None
            Optional identifier; generated automatically if not supplied.

        Returns
        -------
        ExperimentResults
            Fully populated results dataclass.
        """
        if experiment_id is None:
            experiment_id = str(uuid.uuid4())

        results = ExperimentResults(
            experiment_id=experiment_id,
            status="running",
            started_at=datetime.now().isoformat(),
        )

        # Store immediately so status endpoint can find the running experiment
        self.results[experiment_id] = results

        try:
            # 1 – Dataset info
            results.dataset_info = self.get_dataset_info()

            # 2 – Baseline XGBoost (all features)
            results.split_results = self.run_80_20_split(model_type="xgboost")
            results.cv5_results = self.run_cross_validation(n_folds=5, model_type="xgboost")
            results.cv10_results = self.run_cross_validation(n_folds=10, model_type="xgboost")

            # 3 – Genetic Algorithm
            ga_output = self.run_genetic_algorithm()
            results.ga_config = ga_output["ga_config"]
            results.ga_results = ga_output["ga_results"]

            # Derive selected feature indices from the mask
            ga_feature_indices = [
                i for i, v in enumerate(results.ga_results["feature_mask"]) if v == 1
            ]

            # 4 – XGBoost with GA features
            results.ga_xgboost_split_results = self.run_80_20_split(
                model_type="xgboost", feature_indices=ga_feature_indices
            )
            results.ga_xgboost_cv5_results = self.run_cross_validation(
                n_folds=5, model_type="xgboost", feature_indices=ga_feature_indices
            )
            results.ga_xgboost_cv10_results = self.run_cross_validation(
                n_folds=10, model_type="xgboost", feature_indices=ga_feature_indices
            )

            # 5 – Random Forest with GA features
            results.ga_rf_split_results = self.run_80_20_split(
                model_type="random_forest", feature_indices=ga_feature_indices
            )
            results.ga_rf_cv5_results = self.run_cross_validation(
                n_folds=5, model_type="random_forest", feature_indices=ga_feature_indices
            )
            results.ga_rf_cv10_results = self.run_cross_validation(
                n_folds=10, model_type="random_forest", feature_indices=ga_feature_indices
            )

            # 6 – Comparison table
            results.comparison_table = [
                {
                    "Model": "XGBoost (All Features)",
                    "Features": ", ".join(self.feature_names),
                    "N_Features": results.split_results["n_features"],
                    "Split_Accuracy": results.split_results["accuracy"],
                    "Split_F1": results.split_results["f1"],
                    "Split_ROC_AUC": results.split_results["roc_auc"],
                    "CV5_Mean_F1": results.cv5_results["mean_f1"],
                    "CV10_Mean_F1": results.cv10_results["mean_f1"],
                    "Split_Training_Time": results.split_results["training_time"],
                },
                {
                    "Model": "XGBoost (GA Features)",
                    "Features": ", ".join(results.ga_results["selected_features"]),
                    "N_Features": results.ga_xgboost_split_results["n_features"],
                    "Split_Accuracy": results.ga_xgboost_split_results["accuracy"],
                    "Split_F1": results.ga_xgboost_split_results["f1"],
                    "Split_ROC_AUC": results.ga_xgboost_split_results["roc_auc"],
                    "CV5_Mean_F1": results.ga_xgboost_cv5_results["mean_f1"],
                    "CV10_Mean_F1": results.ga_xgboost_cv10_results["mean_f1"],
                    "Split_Training_Time": results.ga_xgboost_split_results["training_time"],
                },
                {
                    "Model": "Random Forest (GA Features)",
                    "Features": ", ".join(results.ga_results["selected_features"]),
                    "N_Features": results.ga_rf_split_results["n_features"],
                    "Split_Accuracy": results.ga_rf_split_results["accuracy"],
                    "Split_F1": results.ga_rf_split_results["f1"],
                    "Split_ROC_AUC": results.ga_rf_split_results["roc_auc"],
                    "CV5_Mean_F1": results.ga_rf_cv5_results["mean_f1"],
                    "CV10_Mean_F1": results.ga_rf_cv10_results["mean_f1"],
                    "Split_Training_Time": results.ga_rf_split_results["training_time"],
                },
            ]

            results.status = "completed"
            results.completed_at = datetime.now().isoformat()

        except Exception as exc:
            results.status = "error"
            results.error_message = str(exc)
            results.completed_at = datetime.now().isoformat()

        # Persist in the in-memory store
        self.results[experiment_id] = results
        return results

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------
    def get_results(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Retrieve results for a given experiment ID, or ``None``."""
        return self.results.get(experiment_id)

    def get_all_experiment_ids(self) -> List[str]:
        """Return a list of all stored experiment IDs."""
        return list(self.results.keys())

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def export_results_json(self, experiment_id: str) -> dict:
        """Convert an ``ExperimentResults`` instance to a JSON-serializable dict.

        Parameters
        ----------
        experiment_id : str
            The experiment to export.

        Returns
        -------
        dict
            Plain Python dict safe for ``json.dumps``.

        Raises
        ------
        ValueError
            If the experiment ID is not found.
        """
        result = self.results.get(experiment_id)
        if result is None:
            raise ValueError(f"Experiment '{experiment_id}' not found.")
        return asdict(result)

    def export_results_csv(self, experiment_id: str) -> str:
        """Return a CSV string of the comparison table.

        Parameters
        ----------
        experiment_id : str
            The experiment to export.

        Returns
        -------
        str
            CSV-formatted string with headers.

        Raises
        ------
        ValueError
            If the experiment ID is not found or has no comparison table.
        """
        result = self.results.get(experiment_id)
        if result is None:
            raise ValueError(f"Experiment '{experiment_id}' not found.")
        if not result.comparison_table:
            raise ValueError(f"Experiment '{experiment_id}' has no comparison table.")

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=result.comparison_table[0].keys())
        writer.writeheader()
        writer.writerows(result.comparison_table)
        return output.getvalue()


# --------------------------------------------------------------------------
# Module-level singleton
# --------------------------------------------------------------------------
experiment_engine = ExperimentEngine()
