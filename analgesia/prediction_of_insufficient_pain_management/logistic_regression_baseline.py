"""
Logistic Regression Baseline Model for Insufficient Pain Management Prediction.

This module implements a logistic regression model as the baseline for predicting
insufficient pain management in prehospital trauma patients.

Author: Generated for ICU Research Project
Date: September 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_validate, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogisticRegressionBaseline:
    """
    Logistic Regression baseline model for insufficient pain management prediction.
    
    This class provides a comprehensive baseline model with:
    - Hyperparameter tuning
    - Cross-validation
    - Comprehensive evaluation metrics
    - Feature importance analysis
    - Model interpretation
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the logistic regression baseline model.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.cv_results = None
        self.feature_names = None
        self.evaluation_results = {}
        
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search with cross-validation.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Best parameters and grid search results
        """
        logger.info("Starting hyperparameter tuning for logistic regression")
        
        # Simplified grid for efficiency
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'max_iter': [1000]
        }
        
        # Create stratified cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Initialize logistic regression
        lr = LogisticRegression(random_state=self.random_state, class_weight='balanced')
        
        # Grid search
        grid_search = GridSearchCV(
            lr, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best cross-validation AUC: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': grid_search.best_score_,
            'grid_search': grid_search
        }
    
    def cross_validate_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation evaluation.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info("Performing cross-validation")
        
        if self.model is None:
            # Use default model if not tuned
            self.model = LogisticRegression(
                random_state=self.random_state, 
                class_weight='balanced',
                max_iter=1000
            )
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Stratified cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        cv_results = cross_validate(
            self.model, X_train, y_train, 
            cv=cv, scoring=scoring, return_train_score=True
        )
        
        # Calculate statistics
        results_summary = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results_summary[metric] = {
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'test_scores': test_scores,
                'train_scores': train_scores
            }
        
        self.cv_results = results_summary
        
        # Log results
        logger.info("Cross-validation results:")
        for metric, stats in results_summary.items():
            logger.info(f"  {metric.upper()}: {stats['test_mean']:.4f} ± {stats['test_std']:.4f}")
        
        return results_summary
    
    def fit_final_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'LogisticRegressionBaseline':
        """
        Fit the final model on the entire training set.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            LogisticRegressionBaseline: Self for method chaining
        """
        logger.info("Fitting final model on entire training set")
        
        if self.model is None:
            self.model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )
        
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist()
        
        logger.info("Final model fitted successfully")
        return self
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the model on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        logger.info("Evaluating model on test data")
        
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
        
        # Precision-Recall curve data
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_curve'] = {'precision': precision, 'recall': recall}
        
        # Calibration curve data
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        metrics['calibration'] = {'prob_true': prob_true, 'prob_pred': prob_pred}
        
        self.evaluation_results = metrics
        
        # Log key metrics
        logger.info("Test set evaluation results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-score: {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the logistic regression coefficients.
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get coefficients
        coefficients = self.model.coef_[0]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def print_model_summary(self) -> None:
        """Print a comprehensive summary of the model and its performance."""
        print("=" * 60)
        print("LOGISTIC REGRESSION BASELINE MODEL SUMMARY")
        print("=" * 60)
        
        if self.best_params:
            print("\\nBest Hyperparameters:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
        
        if self.cv_results:
            print("\\nCross-Validation Results (Mean ± Std):")
            for metric, stats in self.cv_results.items():
                print(f"  {metric.upper()}: {stats['test_mean']:.4f} ± {stats['test_std']:.4f}")
        
        if self.evaluation_results:
            print("\\nTest Set Performance:")
            print(f"  Accuracy: {self.evaluation_results['accuracy']:.4f}")
            print(f"  Precision: {self.evaluation_results['precision']:.4f}")
            print(f"  Recall: {self.evaluation_results['recall']:.4f}")
            print(f"  F1-score: {self.evaluation_results['f1']:.4f}")
            print(f"  ROC-AUC: {self.evaluation_results['roc_auc']:.4f}")
            
            print("\\nConfusion Matrix:")
            print(self.evaluation_results['confusion_matrix'])
        
        if self.model and self.feature_names:
            print("\\nTop 10 Most Important Features:")
            importance_df = self.get_feature_importance()
            for i, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        print("=" * 60)


def run_logistic_regression_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                   y_train: pd.Series, y_test: pd.Series,
                                   tune_hyperparams: bool = True,
                                   cv_folds: int = 5) -> LogisticRegressionBaseline:
    """
    Convenience function to run the complete logistic regression baseline pipeline.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
        tune_hyperparams (bool): Whether to tune hyperparameters
        cv_folds (int): Number of cross-validation folds
        
    Returns:
        LogisticRegressionBaseline: Fitted and evaluated model
    """
    logger.info("Running complete logistic regression baseline pipeline")
    
    # Initialize model
    lr_baseline = LogisticRegressionBaseline()
    
    # Hyperparameter tuning
    if tune_hyperparams:
        lr_baseline.tune_hyperparameters(X_train, y_train, cv_folds)
    
    # Cross-validation
    lr_baseline.cross_validate_model(X_train, y_train, cv_folds)
    
    # Fit final model
    lr_baseline.fit_final_model(X_train, y_train)
    
    # Evaluate on test set
    lr_baseline.evaluate_model(X_test, y_test)
    
    # Print summary
    lr_baseline.print_model_summary()
    
    return lr_baseline


if __name__ == "__main__":
    # Example usage with synthetic data
    from sklearn.datasets import make_classification
    
    # Create synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=10, n_clusters_per_class=1, random_state=42,
        class_sep=0.8, flip_y=0.1
    )
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    # Run baseline
    baseline_model = run_logistic_regression_baseline(
        X_train, X_test, y_train, y_test, tune_hyperparams=True
    )