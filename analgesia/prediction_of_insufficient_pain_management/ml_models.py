"""
Machine Learning Models for Insufficient Pain Management Prediction.

This module implements various machine learning models for predicting
insufficient pain management in prehospital trauma patients, including:
- Random Forest
- XGBoost  
- Support Vector Machine
- Neural Network
- Ensemble methods

Author: Generated for ICU Research Project
Date: September 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import (
    cross_validate, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import XGBoost (optional dependency)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
    logger.info("XGBoost is available")
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost is not available. Install with: pip install xgboost")

class MLModelEvaluator:
    """
    Comprehensive machine learning model evaluator for insufficient pain management prediction.
    
    This class handles multiple ML algorithms with:
    - Hyperparameter tuning
    - Cross-validation
    - Model comparison
    - Feature importance analysis
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ML model evaluator.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_results = {}
        self.test_results = {}
        self.feature_names = None
        
    def get_model_configs(self) -> Dict[str, Dict]:
        """
        Get configuration for all available models.
        
        Returns:
            Dict[str, Dict]: Model configurations with parameters and param grids
        """
        configs = {
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'search_type': 'random'
            },
            
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    random_state=self.random_state
                ),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10]
                },
                'search_type': 'grid'
            },
            
            'svm': {
                'model': SVC(
                    random_state=self.random_state,
                    class_weight='balanced',
                    probability=True
                ),
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                },
                'search_type': 'grid'
            },
            
            'neural_network': {
                'model': MLPClassifier(
                    random_state=self.random_state,
                    max_iter=1000
                ),
                'param_grid': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                },
                'search_type': 'random'
            }
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            configs['xgboost'] = {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    n_jobs=-1
                ),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'search_type': 'random'
            }
        
        return configs
    
    def tune_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   cv_folds: int = 5, n_iter: int = 20) -> Dict[str, Any]:
        """
        Tune hyperparameters for a specific model.
        
        Args:
            model_name (str): Name of the model to tune
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv_folds (int): Number of cross-validation folds
            n_iter (int): Number of iterations for random search
            
        Returns:
            Dict[str, Any]: Tuning results
        """
        logger.info(f"Tuning hyperparameters for {model_name}")
        
        configs = self.get_model_configs()
        if model_name not in configs:
            raise ValueError(f"Model {model_name} not available")
        
        config = configs[model_name]
        model = config['model']
        param_grid = config['param_grid']
        search_type = config['search_type']
        
        # Create stratified cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Choose search strategy
        if search_type == 'random':
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv, 
                scoring='roc_auc', n_jobs=-1, verbose=1,
                random_state=self.random_state
            )
        else:
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Store results
        self.models[model_name] = search.best_estimator_
        self.best_params[model_name] = search.best_params_
        
        logger.info(f"Best parameters for {model_name}: {search.best_params_}")
        logger.info(f"Best CV score for {model_name}: {search.best_score_:.4f}")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'search_object': search
        }
    
    def cross_validate_model(self, model_name: str, X_train: pd.DataFrame, 
                           y_train: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation for a specific model.
        
        Args:
            model_name (str): Name of the model
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info(f"Cross-validating {model_name}")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Run tune_model first.")
        
        model = self.models[model_name]
        
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
            model, X_train, y_train, 
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
        
        self.cv_results[model_name] = results_summary
        
        # Log results
        logger.info(f"Cross-validation results for {model_name}:")
        for metric, stats in results_summary.items():
            logger.info(f"  {metric.upper()}: {stats['test_mean']:.4f} ± {stats['test_std']:.4f}")
        
        return results_summary
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a model on test data.
        
        Args:
            model_name (str): Name of the model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info(f"Evaluating {model_name} on test data")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Run tune_model first.")
        
        model = self.models[model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
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
        
        self.test_results[model_name] = metrics
        
        # Log key metrics
        logger.info(f"Test results for {model_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-score: {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Get feature importance for models that support it.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Optional[pd.DataFrame]: Feature importance dataframe or None
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if self.feature_names is None:
            logger.warning("Feature names not available")
            return None
        
        model = self.models[model_name]
        
        # Check if model has feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_type = 'importance'
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            importance_type = 'coefficient'
        else:
            logger.warning(f"Model {model_name} does not support feature importance")
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            importance_type: importances
        }).sort_values(importance_type, ascending=False)
        
        return importance_df
    
    def run_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series,
                      models_to_run: Optional[List[str]] = None,
                      cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Run all available models through the complete pipeline.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            y_train (pd.Series): Training target
            y_test (pd.Series): Test target
            models_to_run (Optional[List[str]]): List of models to run. If None, run all.
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Dict[str, Any]]: Results for all models
        """
        logger.info("Running all models through complete pipeline")
        
        self.feature_names = X_train.columns.tolist()
        
        # Get available models
        available_models = list(self.get_model_configs().keys())
        
        if models_to_run is None:
            models_to_run = available_models
        else:
            # Validate requested models
            invalid_models = set(models_to_run) - set(available_models)
            if invalid_models:
                logger.warning(f"Invalid models requested: {invalid_models}")
                models_to_run = [m for m in models_to_run if m in available_models]
        
        logger.info(f"Running models: {models_to_run}")
        
        results = {}
        
        for model_name in models_to_run:
            try:
                logger.info(f"\\n{'='*20} {model_name.upper()} {'='*20}")
                
                # Tune hyperparameters
                tune_results = self.tune_model(model_name, X_train, y_train, cv_folds)
                
                # Cross-validation
                cv_results = self.cross_validate_model(model_name, X_train, y_train, cv_folds)
                
                # Fit final model and evaluate on test set
                self.models[model_name].fit(X_train, y_train)
                test_results = self.evaluate_model(model_name, X_test, y_test)
                
                # Get feature importance if available
                feature_importance = self.get_feature_importance(model_name)
                
                # Store all results
                results[model_name] = {
                    'tuning': tune_results,
                    'cross_validation': cv_results,
                    'test_evaluation': test_results,
                    'feature_importance': feature_importance
                }
                
            except Exception as e:
                logger.error(f"Error running {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        logger.info("All models completed")
        return results
    
    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            models_to_ensemble: Optional[List[str]] = None) -> VotingClassifier:
        """
        Create an ensemble model from the best performing individual models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            models_to_ensemble (Optional[List[str]]): Models to include in ensemble
            
        Returns:
            VotingClassifier: Fitted ensemble model
        """
        logger.info("Creating ensemble model")
        
        if models_to_ensemble is None:
            # Use top 3 models based on cross-validation AUC
            if not self.cv_results:
                raise ValueError("No cross-validation results available")
            
            model_scores = {}
            for model_name, results in self.cv_results.items():
                model_scores[model_name] = results['roc_auc']['test_mean']
            
            # Sort by AUC and take top 3
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            models_to_ensemble = [model[0] for model in sorted_models[:3]]
        
        logger.info(f"Creating ensemble with models: {models_to_ensemble}")
        
        # Create ensemble
        estimators = [(name, self.models[name]) for name in models_to_ensemble]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Fit ensemble
        ensemble.fit(X_train, y_train)
        
        # Store ensemble
        self.models['ensemble'] = ensemble
        
        return ensemble
    
    def get_model_comparison_summary(self) -> pd.DataFrame:
        """
        Get a summary comparison of all models.
        
        Returns:
            pd.DataFrame: Comparison summary
        """
        if not self.test_results:
            raise ValueError("No test results available")
        
        summary_data = []
        
        for model_name, results in self.test_results.items():
            if 'error' not in results:
                row = {
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1'],
                    'ROC-AUC': results['roc_auc']
                }
                
                # Add CV results if available
                if model_name in self.cv_results:
                    cv_auc = self.cv_results[model_name]['roc_auc']
                    row['CV_AUC_Mean'] = cv_auc['test_mean']
                    row['CV_AUC_Std'] = cv_auc['test_std']
                
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('ROC-AUC', ascending=False)
        
        return summary_df
    
    def print_final_summary(self) -> None:
        """Print a comprehensive summary of all model results."""
        print("\\n" + "=" * 80)
        print("MACHINE LEARNING MODELS COMPARISON SUMMARY")
        print("=" * 80)
        
        if self.test_results:
            summary_df = self.get_model_comparison_summary()
            print("\\nModel Performance Comparison:")
            print(summary_df.round(4).to_string(index=False))
            
            # Best model
            best_model = summary_df.iloc[0]['Model']
            best_auc = summary_df.iloc[0]['ROC-AUC']
            print(f"\\nBest performing model: {best_model} (ROC-AUC: {best_auc:.4f})")
            
            # Feature importance for best model
            best_importance = self.get_feature_importance(best_model)
            if best_importance is not None:
                print(f"\\nTop 10 features for {best_model}:")
                print(best_importance.head(10).round(4).to_string(index=False))
        
        print("=" * 80)


def run_complete_ml_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           models_to_run: Optional[List[str]] = None,
                           create_ensemble: bool = True,
                           cv_folds: int = 5) -> MLModelEvaluator:
    """
    Run the complete machine learning pipeline.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
        models_to_run (Optional[List[str]]): Models to run
        create_ensemble (bool): Whether to create ensemble model
        cv_folds (int): Number of cross-validation folds
        
    Returns:
        MLModelEvaluator: Fitted evaluator with all results
    """
    logger.info("Starting complete ML pipeline")
    
    # Initialize evaluator
    evaluator = MLModelEvaluator()
    
    # Run all models
    evaluator.run_all_models(
        X_train, X_test, y_train, y_test, 
        models_to_run=models_to_run, cv_folds=cv_folds
    )
    
    # Create ensemble if requested
    if create_ensemble and len(evaluator.models) > 1:
        try:
            evaluator.create_ensemble_model(X_train, y_train)
            evaluator.evaluate_model('ensemble', X_test, y_test)
            logger.info("Ensemble model created and evaluated")
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
    
    # Print summary
    evaluator.print_final_summary()
    
    return evaluator


if __name__ == "__main__":
    # Example usage with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=10, n_clusters_per_class=1, random_state=42,
        class_sep=0.8, flip_y=0.1
    )
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    # Run complete pipeline
    evaluator = run_complete_ml_pipeline(
        X_train, X_test, y_train, y_test,
        models_to_run=['random_forest', 'gradient_boosting'],  # Test with subset
        create_ensemble=True
    )