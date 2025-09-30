# Prediction of Insufficient Pain Management in Prehospital Trauma Patients

## Project Overview

This project develops machine learning models to predict insufficient pain management in prehospital trauma patients. Insufficient pain management is defined as VAS (Visual Analog Scale) on arrival > 3.

The goal is to predict insufficient pain management using variables available in the prehospital setting to enable early intervention and improve patient outcomes.

## Data Source

**Data Path:** `/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/prehospital/analgesia/data/trauma_categories_Rega Pain Study15.09.2025_v2.xlsx`

**Dataset Characteristics:**
- 12,269 total samples
- 147 total features
- Target variable: Insufficient pain management (VAS_on_arrival > 3)
- Target distribution: 77.3% adequate, 22.7% insufficient pain management

## Project Structure

```
prediction_of_insufficient_pain_management/
├── data_exploration.ipynb          # Initial data exploration and analysis
├── data_preprocessing.py           # Data preprocessing pipeline
├── logistic_regression_baseline.py # Logistic regression baseline model
├── ml_models.py                   # Multiple ML models implementation
├── model_comparison.ipynb         # Comprehensive model comparison
├── prediction_pipeline.py         # Production-ready prediction pipeline
├── test_preprocessing.ipynb       # Testing preprocessing functions
└── readme.md                      # This file
```

## Key Features

### 1. Data Preprocessing (`data_preprocessing.py`)
- **PainManagementDataProcessor**: Complete preprocessing pipeline
- Handles missing values using appropriate strategies
- Feature engineering for prehospital variables
- Categorical encoding (one-hot/label encoding)
- Feature scaling (standard/robust/minmax)
- Train/test split with stratification

**Key Functions:**
- `load_and_preprocess_data()`: One-step data loading and preprocessing
- `create_target_variable()`: Creates binary target from VAS scores
- `identify_prehospital_features()`: Identifies relevant prehospital variables
- `handle_missing_values()`: Intelligent missing value imputation
- `engineer_features()`: Creates new features from existing variables

### 2. Logistic Regression Baseline (`logistic_regression_baseline.py`)
- **LogisticRegressionBaseline**: Comprehensive baseline model
- Hyperparameter tuning with grid search
- Cross-validation evaluation
- Feature importance analysis
- Comprehensive performance metrics

**Key Features:**
- Balanced class weights for imbalanced data
- ROC-AUC optimization
- Feature coefficient interpretation
- Calibration analysis

### 3. Machine Learning Models (`ml_models.py`)
- **MLModelEvaluator**: Multi-model comparison framework
- Supports Random Forest, Gradient Boosting, SVM, Neural Networks
- Optional XGBoost support (if installed)
- Ensemble model creation
- Automated model selection

**Available Models:**
- Random Forest Classifier
- Gradient Boosting Classifier  
- Support Vector Machine
- Multi-layer Perceptron (Neural Network)
- XGBoost Classifier (optional)
- Voting Ensemble

### 4. Model Comparison (`model_comparison.ipynb`)
- Side-by-side comparison of all models
- Performance visualization
- Feature importance analysis
- Clinical interpretation of results
- Final recommendations

### 5. Production Pipeline (`prediction_pipeline.py`)
- **InsufficientPainManagementPredictor**: Production-ready pipeline
- Complete training workflow
- Model persistence (save/load)
- Prediction interface
- Input validation
- Performance monitoring

## Usage Examples

### Basic Usage

```python
from data_preprocessing import load_and_preprocess_data
from logistic_regression_baseline import run_logistic_regression_baseline
from ml_models import run_complete_ml_pipeline

# Load and preprocess data
data_path = "path/to/your/data.xlsx"
processed_data, processor = load_and_preprocess_data(data_path)

# Prepare train/test split
X_train, X_test, y_train, y_test = processor.prepare_modeling_data()

# Run logistic regression baseline
lr_model = run_logistic_regression_baseline(
    X_train, X_test, y_train, y_test, 
    tune_hyperparams=True
)

# Run multiple ML models
ml_evaluator = run_complete_ml_pipeline(
    X_train, X_test, y_train, y_test,
    models_to_run=['random_forest', 'xgboost', 'svm'],
    create_ensemble=True
)
```

### Production Pipeline

```python
from prediction_pipeline import InsufficientPainManagementPredictor

# Initialize predictor
predictor = InsufficientPainManagementPredictor()

# Train with automatic model selection
results = predictor.train_pipeline(
    data_path="path/to/data.xlsx",
    model_type='auto'  # Automatically selects best model
)

# Save trained model
model_path = predictor.save_model()

# Make predictions on new data
sample_patient = {
    'GCS': 14,
    'HR': 85,
    'SPO2': 98,
    'VAS_on_scene': 7
}

prediction = predictor.predict(sample_patient)
print(f"Prediction: {prediction['prediction_labels'][0]}")
print(f"Probability: {prediction['probabilities'][0]:.3f}")
```

## Model Performance

### Evaluation Metrics
All models are evaluated using:
- **ROC-AUC**: Primary metric for model selection
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correctness

### Cross-Validation
- 5-fold stratified cross-validation
- Maintains class distribution across folds
- Reports mean ± standard deviation for all metrics

### Clinical Interpretation
- **High Precision**: Fewer false alarms for insufficient pain management
- **High Recall**: Better identification of patients needing intervention
- **Calibration**: Probability scores reflect true likelihood

## Prehospital Variables

The models use variables available in the prehospital setting:

### Demographics & Vitals
- Age, Gender
- Heart Rate (HR)
- Blood Pressure (Systolic/Diastolic)
- Oxygen Saturation (SPO2)

### Neurological Assessment
- Glasgow Coma Scale (GCS)
- Consciousness Level

### Pain Assessment
- VAS on Scene (initial pain score)
- Pain characteristics

### Injury & Scene Factors
- Mechanism of injury
- Transport duration
- Scene location
- Initial interventions

## Installation & Requirements

### Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl joblib
```

### Optional Packages
```bash
pip install xgboost  # For XGBoost model support
```

### Environment Setup
```python
# Configure environment
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
import numpy as np
np.random.seed(42)
```

## Model Development Workflow

1. **Data Exploration** (`data_exploration.ipynb`)
   - Understand data structure and quality
   - Identify target variable distribution
   - Explore prehospital variables

2. **Preprocessing** (`data_preprocessing.py`)
   - Clean and prepare data
   - Handle missing values
   - Engineer relevant features

3. **Baseline Model** (`logistic_regression_baseline.py`)
   - Establish performance baseline
   - Feature importance analysis
   - Model interpretability

4. **Advanced Models** (`ml_models.py`)
   - Compare multiple algorithms
   - Hyperparameter optimization
   - Ensemble methods

5. **Model Selection** (`model_comparison.ipynb`)
   - Comprehensive comparison
   - Performance visualization
   - Clinical validation

6. **Production Deployment** (`prediction_pipeline.py`)
   - Production-ready pipeline
   - Model persistence
   - Prediction interface

## Best Practices

### Data Handling
- Always use stratified splits for imbalanced data
- Apply same preprocessing to train/test sets
- Handle missing values appropriately
- Validate feature consistency

### Model Training
- Use cross-validation for reliable estimates
- Optimize for clinically relevant metrics
- Consider class imbalance in model selection
- Validate on hold-out test set

### Clinical Implementation
- Interpret probability scores carefully
- Consider false positive/negative costs
- Validate on external datasets
- Monitor model performance over time

## Future Enhancements

### Data Improvements
- Additional prehospital variables
- External validation datasets
- Temporal analysis of pain progression
- Multi-center validation

### Model Enhancements
- Deep learning approaches
- Time-series modeling for dynamic assessment
- Uncertainty quantification
- Explainable AI techniques

### Clinical Integration
- Real-time prediction interface
- Clinical decision support integration
- Outcome tracking and feedback
- Continuous model updating

## Contributing

When contributing to this project:

1. Follow the established code structure
2. Document all functions comprehensively
3. Use consistent naming conventions
4. Include proper error handling
5. Add unit tests for new functionality
6. Update this README for significant changes

## Contact & Support

For questions about this project:
- Review the comprehensive documentation in each module
- Check the example notebooks for usage patterns
- Refer to the model comparison results for performance insights

## License & Ethics

This project is developed for research purposes in improving prehospital pain management. When using these models:

- Ensure proper validation before clinical use
- Consider ethical implications of automated predictions
- Maintain patient privacy and data security
- Follow institutional review board guidelines
- Report any issues or biases discovered

---

**Note**: This project provides research tools for pain management prediction. Clinical implementation requires additional validation, regulatory approval, and integration with existing healthcare systems.
