"""
Feature Importance Analysis for XGBoost Insufficient Pain Management Prediction

This script generates visualizations of the most important predictors from the XGBoost model
for predicting insufficient pain management in prehospital trauma care.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path
sys.path.append('/Users/jk1/icu_research/PreHosp')

from analgesia.prediction_of_insufficient_pain_management.data_preprocessing import load_and_preprocess_data
from analgesia.prediction_of_insufficient_pain_management.ml_models import MLModelEvaluator

def create_feature_importance_figure(save_path: str = None):
    """
    Create a comprehensive figure showing XGBoost feature importance for pain management prediction.
    
    Args:
        save_path (str): Path to save the figure. If None, displays the figure.
    """
    print("🚀 Starting Feature Importance Analysis for XGBoost Model")
    print("=" * 60)
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    data_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/prehospital/analgesia/data/trauma_categories_Rega Pain Study15.09.2025_v2.xlsx'
    processed_data, processor = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = processor.prepare_modeling_data()
    
    print(f"✅ Data loaded: {X_train.shape[0]} training samples, {X_train.shape[1]} features")
    
    # Train XGBoost model
    print("🔄 Training XGBoost model with hyperparameter tuning...")
    evaluator = MLModelEvaluator()
    evaluator.tune_model('xgboost', X_train, y_train)
    
    xgb_model = evaluator.models['xgboost']
    best_params = evaluator.best_params.get('xgboost', {})
    
    # Get predictions and performance
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"✅ Model trained - Accuracy: {accuracy:.3f}, ROC-AUC: {roc_auc:.3f}")
    
    # Extract feature importance
    feature_importance = xgb_model.feature_importances_
    feature_names = X_train.columns.tolist()
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Add clinical interpretation
    clinical_mapping = {
        'VAS_on_scene': 'VAS Pain Score at Scene',
        'GCS': 'Glasgow Coma Scale',
        'HR': 'Heart Rate',
        'SPO2': 'Oxygen Saturation',
        'Bewusstseinlage': 'Consciousness Level',
        'HR5': 'Heart Rate (5 min)',
        'GCS7': 'Glasgow Coma Scale (7 min)',
        'SPO211': 'Oxygen Saturation (11 min)',
        'Lagerungen': 'Patient Positioning',
        'Abfahrtsort': 'Departure Location',
        'Geschlecht_Weiblich': 'Female Gender',
        'Geschlecht_Unbekannt': 'Unknown Gender',
        'HR_category_Normal': 'Normal Heart Rate',
        'HR_category_Tachycardia': 'Tachycardia',
        'HR_category_Severe_Tachycardia': 'Severe Tachycardia',
        'SPO2_category_Normal': 'Normal Oxygen Saturation',
        'SPO2_category_Severe_Hypoxia': 'Severe Hypoxia',
        'Ist Reanimation durchgeführt_Nein': 'No Resuscitation'
    }
    
    # Map feature names to clinical descriptions
    importance_df['clinical_name'] = importance_df['feature'].map(
        lambda x: clinical_mapping.get(x, x.replace('Thoraxdrainage_', 'Chest Tube: ').replace(', 0', ''))
    )
    
    # Create the figure
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('XGBoost Feature Importance Analysis\nPredicting Insufficient Pain Management in Prehospital Trauma Care', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    
    # 1. Top 15 Feature Importance Bar Plot
    top_features = importance_df.head(15)
    bars1 = ax1.barh(range(len(top_features)), top_features['importance'], color=colors[:15])
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['clinical_name'], fontsize=10)
    ax1.set_xlabel('Feature Importance', fontweight='bold')
    ax1.set_title('Top 15 Most Important Predictors', fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Add importance values on bars
    for i, (bar, importance) in enumerate(zip(bars1, top_features['importance'])):
        ax1.text(importance + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center', fontsize=8, fontweight='bold')
    
    # 2. Feature Importance by Category
    categories = {
        'Vital Signs': ['HR', 'SPO2', 'HR5', 'SPO211', 'HR_category', 'SPO2_category'],
        'Neurological': ['GCS', 'Bewusstseinlage', 'GCS7'],
        'Pain Assessment': ['VAS_on_scene'],
        'Demographics': ['Geschlecht'],
        'Scene/Transport': ['Abfahrtsort', 'Lagerungen', 'Ist Reanimation'],
        'Medical Interventions': ['Thoraxdrainage']
    }
    
    category_importance = {}
    for category, patterns in categories.items():
        total_importance = 0
        for _, row in importance_df.iterrows():
            for pattern in patterns:
                if pattern in row['feature']:
                    total_importance += row['importance']
                    break
        category_importance[category] = total_importance
    
    cat_df = pd.DataFrame(list(category_importance.items()), columns=['Category', 'Importance'])
    cat_df = cat_df.sort_values('Importance', ascending=True)
    
    bars2 = ax2.barh(cat_df['Category'], cat_df['Importance'], 
                     color=plt.cm.Set3(np.linspace(0, 1, len(cat_df))))
    ax2.set_xlabel('Cumulative Importance', fontweight='bold')
    ax2.set_title('Feature Importance by Clinical Category', fontweight='bold', pad=20)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for bar, importance in zip(bars2, cat_df['Importance']):
        ax2.text(importance + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center', fontweight='bold')
    
    # 3. Cumulative Importance
    cumulative_importance = np.cumsum(importance_df['importance'])
    ax3.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
             marker='o', linewidth=2, markersize=4, color='darkblue')
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% of total importance')
    ax3.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% of total importance')
    ax3.set_xlabel('Number of Features', fontweight='bold')
    ax3.set_ylabel('Cumulative Importance', fontweight='bold')
    ax3.set_title('Cumulative Feature Importance', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, len(cumulative_importance))
    ax3.set_ylim(0, 1.05)
    
    # Find features needed for 80% and 90% importance
    features_80 = np.argmax(cumulative_importance >= 0.8) + 1
    features_90 = np.argmax(cumulative_importance >= 0.9) + 1
    ax3.axvline(x=features_80, color='red', linestyle=':', alpha=0.7)
    ax3.axvline(x=features_90, color='orange', linestyle=':', alpha=0.7)
    ax3.text(features_80 + 0.5, 0.85, f'{features_80} features\n(80%)', fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax3.text(features_90 + 0.5, 0.95, f'{features_90} features\n(90%)', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. Model Performance Summary
    ax4.axis('off')
    performance_text = f"""
    Model Performance Summary
    
    Algorithm: XGBoost Classifier
    Training Samples: {X_train.shape[0]:,}
    Test Samples: {X_test.shape[0]:,}
    Total Features: {X_train.shape[1]}
    
    Best Hyperparameters:
    • Learning Rate: {best_params.get('learning_rate', 'N/A')}
    • Max Depth: {best_params.get('max_depth', 'N/A')}
    • N Estimators: {best_params.get('n_estimators', 'N/A')}
    • Subsample: {best_params.get('subsample', 'N/A')}
    • Colsample Bytree: {best_params.get('colsample_bytree', 'N/A')}
    
    Test Performance:
    • Accuracy: {accuracy:.1%}
    • ROC-AUC: {roc_auc:.1%}
    
    Top 3 Predictors:
    1. {top_features.iloc[0]['clinical_name']} ({top_features.iloc[0]['importance']:.3f})
    2. {top_features.iloc[1]['clinical_name']} ({top_features.iloc[1]['importance']:.3f})
    3. {top_features.iloc[2]['clinical_name']} ({top_features.iloc[2]['importance']:.3f})
    
    Clinical Insight:
    • {features_80} features capture 80% of predictive power
    • {features_90} features capture 90% of predictive power
    • Model uses only prehospital variables
    • No data leakage from hospital outcomes
    """
    
    ax4.text(0.05, 0.95, performance_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Figure saved to: {save_path}")
    else:
        plt.show()
    
    return importance_df, fig

def print_feature_importance_summary(importance_df):
    """Print a detailed summary of feature importance."""
    print("\n" + "="*70)
    print("📋 FEATURE IMPORTANCE DETAILED SUMMARY")
    print("="*70)
    
    print(f"\n🎯 Top 10 Most Important Predictors:")
    print("-" * 50)
    for i, row in importance_df.head(10).iterrows():
        print(f"{i+1:2d}. {row['clinical_name']:<35} {row['importance']:.4f}")
    
    # Calculate cumulative importance milestones
    cumulative = np.cumsum(importance_df['importance'])
    features_50 = np.argmax(cumulative >= 0.5) + 1
    features_80 = np.argmax(cumulative >= 0.8) + 1
    features_90 = np.argmax(cumulative >= 0.9) + 1
    
    print(f"\n📊 Cumulative Importance Milestones:")
    print(f"   • Top {features_50} features capture 50% of importance")
    print(f"   • Top {features_80} features capture 80% of importance") 
    print(f"   • Top {features_90} features capture 90% of importance")
    
    print(f"\n🏥 Clinical Categories (by total importance):")
    categories = {
        'Vital Signs': ['HR', 'SPO2'],
        'Neurological': ['GCS', 'Bewusstseinlage'],
        'Pain Assessment': ['VAS_on_scene'],
        'Demographics': ['Geschlecht'],
        'Scene/Transport': ['Abfahrtsort', 'Lagerungen', 'Ist Reanimation'],
        'Medical Interventions': ['Thoraxdrainage']
    }
    
    for category, patterns in categories.items():
        total_importance = sum(row['importance'] for _, row in importance_df.iterrows() 
                             if any(pattern in row['feature'] for pattern in patterns))
        print(f"   • {category:<20} {total_importance:.4f}")

if __name__ == "__main__":
    # Create the feature importance figure
    save_path = "/Users/jk1/icu_research/PreHosp/analgesia/prediction_of_insufficient_pain_management/xgboost_feature_importance.png"
    
    try:
        importance_df, fig = create_feature_importance_figure(save_path=save_path)
        print_feature_importance_summary(importance_df)
        
        print(f"\n🎊 Feature importance analysis completed successfully!")
        print(f"📁 Figure saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Error during feature importance analysis: {str(e)}")
        import traceback
        traceback.print_exc()