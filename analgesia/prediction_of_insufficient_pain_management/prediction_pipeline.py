"""
Production-Ready Pain Management Prediction Pipeline.

This module provides a complete, production-ready pipeline for predicting
insufficient pain management in prehospital trauma patients.

Features:
- Complete preprocessing pipeline
- Model selection and training
- Prediction interface
- Model persistence (save/load)
- Input validation
- Comprehensive logging

Author: Generated for ICU Research Project
Date: September 2025
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_preprocessing import PainManagementDataProcessor
from logistic_regression_baseline import LogisticRegressionBaseline
from ml_models import MLModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsufficientPainManagementPredictor:
    """
    Production-ready predictor for insufficient pain management.
    
    This class provides a complete pipeline for:
    - Data preprocessing
    - Model training and selection
    - Making predictions
    - Model persistence
    - Performance monitoring
    """
    
    def __init__(self, model_save_dir: str = "models"):
        """
        Initialize the predictor.
        
        Args:
            model_save_dir (str): Directory to save/load models
        """
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        # Pipeline components
        self.preprocessor = None
        self.model = None
        self.model_type = None
        self.feature_names = None
        self.performance_metrics = None
        self.training_timestamp = None
        
        # Model metadata
        self.metadata = {
            'version': '1.0.0',
            'target_definition': 'VAS_on_arrival > 3',
            'features_used': None,
            'training_samples': None,
            'model_performance': None
        }
    
    def train_pipeline(self, data_path: str, 
                      model_type: str = 'auto',
                      test_size: float = 0.2,
                      cv_folds: int = 5,
                      random_state: int = 42) -> Dict[str, Any]:
        """
        Train the complete pipeline from raw data.
        
        Args:
            data_path (str): Path to training data
            model_type (str): Type of model ('logistic', 'random_forest', 'xgboost', 'auto')
            test_size (float): Proportion for test set
            cv_folds (int): Cross-validation folds
            random_state (int): Random state for reproducibility
            
        Returns:
            Dict[str, Any]: Training results and performance metrics
        """
        logger.info("Starting complete pipeline training")
        self.training_timestamp = datetime.now()
        
        # Step 1: Load and preprocess data
        logger.info("Step 1: Data preprocessing")
        self.preprocessor = PainManagementDataProcessor(data_path)
        processed_data = self.preprocessor.run_full_pipeline()
        
        # Step 2: Prepare modeling data
        logger.info("Step 2: Preparing train/test split")
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_modeling_data(
            test_size=test_size, random_state=random_state
        )
        
        self.feature_names = X_train.columns.tolist()
        
        # Step 3: Model selection and training
        logger.info("Step 3: Model training and selection")
        
        if model_type == 'auto':
            # Run model comparison to select best
            results = self._run_model_comparison(X_train, X_test, y_train, y_test, cv_folds)
            best_model_name = results['best_model_name']
            self.model = results['best_model']
            self.model_type = best_model_name
            self.performance_metrics = results['performance_metrics']
        elif model_type == 'logistic':
            # Train logistic regression
            lr_baseline = LogisticRegressionBaseline(random_state=random_state)
            lr_baseline.tune_hyperparameters(X_train, y_train, cv_folds)
            lr_baseline.cross_validate_model(X_train, y_train, cv_folds)
            lr_baseline.fit_final_model(X_train, y_train)
            lr_baseline.evaluate_model(X_test, y_test)
            
            self.model = lr_baseline
            self.model_type = 'logistic_regression'
            self.performance_metrics = lr_baseline.evaluation_results
            
        else:
            # Train specific ML model
            ml_evaluator = MLModelEvaluator(random_state=random_state)
            ml_evaluator.tune_model(model_type, X_train, y_train, cv_folds)
            ml_evaluator.cross_validate_model(model_type, X_train, y_train, cv_folds)
            ml_evaluator.models[model_type].fit(X_train, y_train)
            ml_evaluator.evaluate_model(model_type, X_test, y_test)
            
            self.model = ml_evaluator.models[model_type]
            self.model_type = model_type
            self.performance_metrics = ml_evaluator.test_results[model_type]
        
        # Step 4: Update metadata
        self.metadata.update({
            'features_used': self.feature_names,
            'training_samples': len(X_train),
            'model_performance': {
                'roc_auc': self.performance_metrics['roc_auc'],
                'precision': self.performance_metrics['precision'],
                'recall': self.performance_metrics['recall'],
                'f1': self.performance_metrics['f1']
            },
            'training_date': self.training_timestamp.isoformat()
        })
        
        logger.info(f"Pipeline training completed. Best model: {self.model_type}")
        logger.info(f"Performance - ROC-AUC: {self.performance_metrics['roc_auc']:.4f}")
        
        return {
            'model_type': self.model_type,
            'performance_metrics': self.performance_metrics,
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }