"""
Simplified Production Pipeline Test for Insufficient Pain Management Prediction.

This notebook tests a simplified version of the production pipeline.
"""

import sys
import os
sys.path.append('/Users/jk1/icu_research/PreHosp')

from analgesia.prediction_of_insufficient_pain_management.data_preprocessing import load_and_preprocess_data, PainManagementDataProcessor
from analgesia.prediction_of_insufficient_pain_management.ml_models import MLModelEvaluator
import pandas as pd
import numpy as np
import joblib
import tempfile
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class SimplifiedProductionPredictor:
    """Simplified production predictor for testing purposes."""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.feature_names = None
        self.performance_metrics = None
        self.training_timestamp = None
    
    def train(self, X_train, y_train, model_type='random_forest'):
        """Train a specific model type."""
        print(f"Training {model_type} model...")
        
        # Initialize ML evaluator
        ml_evaluator = MLModelEvaluator()
        
        # Tune and train the specified model
        ml_evaluator.tune_model(model_type, X_train, y_train)
        
        # Store trained model
        self.model = ml_evaluator.models[model_type]
        self.model_type = model_type
        self.feature_names = list(X_train.columns)
        self.training_timestamp = datetime.now()
        
        print(f"Training completed for {model_type}")
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("No model trained. Call train() first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return self.performance_metrics
    
    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("No model trained. Call train() first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        save_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'training_timestamp': self.training_timestamp
        }
        
        joblib.dump(save_data, filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """Load a saved model."""
        save_data = joblib.load(filepath)
        
        self.model = save_data['model']
        self.model_type = save_data['model_type']
        self.feature_names = save_data['feature_names']
        self.performance_metrics = save_data['performance_metrics']
        self.training_timestamp = save_data['training_timestamp']
        
        print(f"Model loaded: {self.model_type}")
        return self

if __name__ == "__main__":
    print("Simplified Production Pipeline Test")
    print("=" * 40)
    
    try:
        # Load and prepare data
        print("\n1. Loading and preparing data...")
        # Use the actual data path and convenience function like the notebooks do
        data_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/prehospital/analgesia/data/trauma_categories_Rega Pain Study15.09.2025_v2.xlsx'
        
        # Use the load_and_preprocess_data function like the notebooks
        processed_data, processor = load_and_preprocess_data(data_path)
        X_train, X_test, y_train, y_test = processor.prepare_modeling_data()
        
        print(f"Data loaded: {X_train.shape[0] + X_test.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Target distribution: {pd.Series(list(y_train) + list(y_test)).value_counts().to_dict()}")
        
        # For production test, we'll just use the built-in train/test split
        # No need to split again
        print("\n2. Using built-in train/test split...")
        
        # Initialize and train production predictor
        print("\n3. Training production model...")
        predictor = SimplifiedProductionPredictor()
        
        # Use XGBoost as the default since it's now the best performer
        model_type = 'xgboost'  # Changed from 'random_forest' to 'xgboost'
        predictor.train(X_train, y_train, model_type=model_type)
        
        # Make predictions
        print("\n4. Making predictions...")
        y_pred, y_prob = predictor.predict(X_test)
        
        # Calculate performance
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test ROC-AUC: {roc_auc:.4f}")
        
        # Test model saving and loading
        print("\n5. Testing model persistence...")
        model_path = "test_production_model.joblib"
        predictor.save_model(model_path)
        
        # Create new predictor and load model
        new_predictor = SimplifiedProductionPredictor()
        new_predictor.load_model(model_path)
        
        # Test loaded model
        y_pred_loaded, y_prob_loaded = new_predictor.predict(X_test[:5])
        print(f"Loaded model predictions on 5 samples: {y_pred_loaded}")
        
        # Test single sample prediction
        print("\n6. Testing single sample prediction...")
        single_sample = X_test.iloc[[0]]
        pred, prob = new_predictor.predict(single_sample)
        print(f"Single sample prediction: {pred[0]} (probability: {prob[0]:.4f})")
        
        print("\n✅ Production pipeline test completed successfully!")
        
        # Clean up
        import os
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Cleaned up: {model_path}")
            
    except Exception as e:
        print(f"\n❌ Production pipeline test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()