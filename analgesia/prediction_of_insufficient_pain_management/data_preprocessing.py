"""
Data preprocessing functions for insufficient pain management prediction.

This module contains functions for:
- Loading and cleaning the dataset
- Feature engineering
- Handling missing values
- Preparing prehospital variables for modeling
- Creating train/test splits

Author: Generated for ICU Research Project
Date: September 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PainManagementDataProcessor:
    """
    Main class for processing pain management prediction data.
    
    This class handles data loading, cleaning, feature engineering, and preprocessing
    for predicting insufficient pain management based on prehospital variables.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data processor.
        
        Args:
            data_path (str): Path to the Excel data file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = None
        self.target_column = 'insufficient_pain_mgmt'
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the Excel data file.
        
        Returns:
            pd.DataFrame: Raw data loaded from Excel file
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            Exception: If there's an error reading the file
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.raw_data = pd.read_excel(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
            return self.raw_data
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the target variable for insufficient pain management.
        
        Insufficient pain management is defined as VAS_on_arrival > 3.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with target variable added
        """
        logger.info("Creating target variable: insufficient_pain_mgmt")
        
        # Create binary target variable
        df[self.target_column] = (df['VAS_on_arrival'] > 3).astype(int)
        
        # Log target distribution
        target_counts = df[self.target_column].value_counts()
        logger.info(f"Target distribution - Adequate: {target_counts.get(0, 0)}, "
                   f"Insufficient: {target_counts.get(1, 0)}")
        
        return df
    
    def identify_prehospital_features(self, df: pd.DataFrame) -> List[str]:
        """
        Identify potential prehospital features available for prediction.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            List[str]: List of potential prehospital feature column names
        """
        logger.info("Identifying prehospital features")
        
        # Define categories of prehospital variables
        prehospital_patterns = {
            'demographics': ['age', 'geschlecht', 'sex', 'weight', 'height', 'bmi'],
            'vital_signs': ['hr', 'bp', 'systolic', 'diastolic', 'spo2', 'oxygen', 'resp'],
            'neurological': ['gcs', 'bewusst', 'consciousness', 'alert'],
            'injury_mechanism': ['mechanism', 'trauma', 'injury', 'accident', 'fall'],
            'scene_factors': ['scene', 'location', 'time', 'transport', 'duration'],
            'initial_assessment': ['vas_on_scene', 'pain', 'severity']
        }
        
        potential_features = []
        
        # Search for columns matching prehospital patterns
        for category, patterns in prehospital_patterns.items():
            for col in df.columns:
                col_lower = col.lower()
                for pattern in patterns:
                    if pattern in col_lower and col not in potential_features:
                        potential_features.append(col)
                        break
        
        # Manually add known important prehospital variables
        known_prehospital = [
            'GCS', 'HR', 'SPO2', 'VAS_on_scene', 'Bewusstseinlage',
            'Lagerungen', 'Atemwegmanagement', 'Abfahrtsort'
        ]
        
        for col in known_prehospital:
            if col in df.columns and col not in potential_features:
                potential_features.append(col)
        
        logger.info(f"Identified {len(potential_features)} potential prehospital features")
        return potential_features
    
    def clean_data(self, df: pd.DataFrame, 
                   remove_high_missing_threshold: float = 0.8) -> pd.DataFrame:
        """
        Clean the dataset by removing columns with excessive missing data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            remove_high_missing_threshold (float): Threshold for removing columns with 
                                                 missing data (default: 0.8)
        
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Cleaning data")
        
        initial_shape = df.shape
        
        # Calculate missing data percentage
        missing_pct = df.isnull().sum() / len(df)
        
        # Remove columns with excessive missing data
        cols_to_remove = missing_pct[missing_pct > remove_high_missing_threshold].index
        logger.info(f"Removing {len(cols_to_remove)} columns with >{remove_high_missing_threshold*100}% missing data")
        
        df_cleaned = df.drop(columns=cols_to_remove)
        
        # Remove completely empty columns
        empty_cols = df_cleaned.columns[df_cleaned.isnull().all()].tolist()
        if empty_cols:
            logger.info(f"Removing {len(empty_cols)} completely empty columns")
            df_cleaned = df_cleaned.drop(columns=empty_cols)
        
        logger.info(f"Data cleaning complete. Shape: {initial_shape} -> {df_cleaned.shape}")
        return df_cleaned
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values using different strategies for different variable types.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (Dict[str, str]): Strategy for handling missing values by column type
                                     Default: {'numeric': 'median', 'categorical': 'most_frequent'}
        
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        if strategy is None:
            strategy = {'numeric': 'median', 'categorical': 'most_frequent'}
        
        logger.info("Handling missing values")
        df_imputed = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numeric missing values
        if numeric_cols:
            numeric_imputer = SimpleImputer(strategy=strategy.get('numeric', 'median'))
            df_imputed[numeric_cols] = numeric_imputer.fit_transform(df_imputed[numeric_cols])
            self.imputers['numeric'] = numeric_imputer
            logger.info(f"Imputed {len(numeric_cols)} numeric columns")
        
        # Handle categorical missing values
        if categorical_cols:
            categorical_imputer = SimpleImputer(strategy=strategy.get('categorical', 'most_frequent'))
            df_imputed[categorical_cols] = categorical_imputer.fit_transform(df_imputed[categorical_cols])
            self.imputers['categorical'] = categorical_imputer
            logger.info(f"Imputed {len(categorical_cols)} categorical columns")
        
        return df_imputed
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        logger.info("Engineering features")
        df_engineered = df.copy()
        
        # Create age groups if age is available
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        if age_cols:
            age_col = age_cols[0]
            try:
                # Convert to numeric, coercing errors to NaN
                age_numeric = pd.to_numeric(df_engineered[age_col], errors='coerce')
                # Only create age groups if we have valid numeric data
                if age_numeric.notna().sum() > 0:
                    df_engineered[f'{age_col}_group'] = pd.cut(
                        age_numeric, 
                        bins=[0, 18, 35, 50, 65, 100], 
                        labels=['<18', '18-35', '36-50', '51-65', '>65']
                    )
                else:
                    logger.warning(f"No valid numeric age data found in column {age_col}")
            except Exception as e:
                logger.warning(f"Could not create age groups from column {age_col}: {str(e)}")
        
        # Create GCS severity categories if GCS is available
        if 'GCS' in df.columns:
            try:
                gcs_numeric = pd.to_numeric(df_engineered['GCS'], errors='coerce')
                if gcs_numeric.notna().sum() > 0:
                    df_engineered['GCS_severity'] = pd.cut(
                        gcs_numeric,
                        bins=[0, 8, 12, 15],
                        labels=['Severe', 'Moderate', 'Mild']
                    )
            except Exception as e:
                logger.warning(f"Could not create GCS severity categories: {str(e)}")
        
        # Create vital signs categories
        if 'HR' in df.columns:
            try:
                hr_numeric = pd.to_numeric(df_engineered['HR'], errors='coerce')
                if hr_numeric.notna().sum() > 0:
                    df_engineered['HR_category'] = pd.cut(
                        hr_numeric,
                        bins=[0, 60, 100, 120, 200],
                        labels=['Bradycardia', 'Normal', 'Tachycardia', 'Severe_Tachycardia']
                    )
            except Exception as e:
                logger.warning(f"Could not create HR categories: {str(e)}")
        
        if 'SPO2' in df.columns:
            try:
                spo2_numeric = pd.to_numeric(df_engineered['SPO2'], errors='coerce')
                if spo2_numeric.notna().sum() > 0:
                    df_engineered['SPO2_category'] = pd.cut(
                        spo2_numeric,
                        bins=[0, 90, 95, 100],
                        labels=['Severe_Hypoxia', 'Mild_Hypoxia', 'Normal']
                    )
            except Exception as e:
                logger.warning(f"Could not create SPO2 categories: {str(e)}")
        
        # Create pain score difference if both scene and arrival VAS are available
        if 'VAS_on_scene' in df.columns and 'VAS_on_arrival' in df.columns:
            try:
                vas_scene_numeric = pd.to_numeric(df_engineered['VAS_on_scene'], errors='coerce')
                vas_arrival_numeric = pd.to_numeric(df_engineered['VAS_on_arrival'], errors='coerce')
                
                if vas_scene_numeric.notna().sum() > 0 and vas_arrival_numeric.notna().sum() > 0:
                    df_engineered['VAS_change'] = vas_arrival_numeric - vas_scene_numeric
                    df_engineered['VAS_improved'] = (df_engineered['VAS_change'] < 0).astype(int)
            except Exception as e:
                logger.warning(f"Could not create VAS change features: {str(e)}")
        
        logger.info(f"Feature engineering complete. Added {len(df_engineered.columns) - len(df.columns)} new features")
        return df_engineered
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                   encoding_method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables for machine learning.
        
        Args:
            df (pd.DataFrame): Input dataframe
            encoding_method (str): Method for encoding ('onehot' or 'label')
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        logger.info(f"Encoding categorical variables using {encoding_method}")
        df_encoded = df.copy()
        
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            logger.info("No categorical columns to encode")
            return df_encoded
        
        if encoding_method == 'onehot':
            # Use one-hot encoding for categorical variables
            for col in categorical_cols:
                if df_encoded[col].nunique() <= 10:  # Only for low cardinality
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded.drop(columns=[col], inplace=True)
                else:
                    # Use label encoding for high cardinality
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le
        
        elif encoding_method == 'label':
            # Use label encoding for all categorical variables
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
        
        logger.info(f"Categorical encoding complete. Shape: {df_encoded.shape}")
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, 
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Scaling method ('standard', 'minmax', or 'robust')
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        logger.info(f"Scaling features using {method} scaling")
        df_scaled = df.copy()
        
        # Get numeric columns (excluding target if present)
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        
        if not numeric_cols:
            logger.info("No numeric columns to scale")
            return df_scaled
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}. Using standard scaling.")
            scaler = StandardScaler()
        
        # Scale the features
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        self.scalers['features'] = scaler
        
        logger.info(f"Feature scaling complete for {len(numeric_cols)} columns")
        return df_scaled
    
    def prepare_modeling_data(self, test_size: float = 0.2, 
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                           pd.Series, pd.Series]:
        """
        Prepare data for modeling by creating train/test splits.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run the full pipeline first.")
        
        logger.info("Preparing data for modeling")
        
        # Remove cases with missing target variable
        complete_cases = self.processed_data[self.target_column].notna()
        data_complete = self.processed_data[complete_cases].copy()
        
        # Separate features and target - exclude VAS_on_arrival and derived features to prevent data leakage
        columns_to_exclude = [
            self.target_column, 
            'VAS_on_arrival',     # Direct leakage - used to create target
            'VAS_change',         # Indirect leakage - contains VAS_on_arrival info
            'VAS_improved'        # Indirect leakage - contains VAS_on_arrival info
        ]
        feature_columns = [col for col in data_complete.columns if col not in columns_to_exclude]
        X = data_complete[feature_columns]
        y = data_complete[self.target_column]
        
        logger.info("Excluded VAS_on_arrival, VAS_change, and VAS_improved from features to prevent data leakage")
        logger.info(f"Using {len(feature_columns)} features for modeling")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        logger.info("Data split complete:")
        logger.info(f"  Training set: {X_train.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        logger.info(f"  Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def run_full_pipeline(self, **kwargs) -> pd.DataFrame:
        """
        Run the complete data preprocessing pipeline.
        
        Args:
            **kwargs: Additional arguments for pipeline steps
            
        Returns:
            pd.DataFrame: Fully processed data ready for modeling
        """
        logger.info("Starting full data preprocessing pipeline")
        
        # Load data
        if self.raw_data is None:
            self.load_data()
        
        # Start with raw data
        df = self.raw_data.copy()
        
        # Create target variable
        df = self.create_target_variable(df)
        
        # Identify prehospital features
        self.feature_columns = self.identify_prehospital_features(df)
        
        # Keep only relevant columns (prehospital features + target + some key columns)
        key_columns = ['VAS_on_arrival', 'VAS_on_scene'] + self.feature_columns + [self.target_column]
        # Remove duplicates while preserving order
        available_columns = []
        for col in key_columns:
            if col in df.columns and col not in available_columns:
                available_columns.append(col)
        df = df[available_columns]
        
        # Clean data
        df = self.clean_data(df, **kwargs.get('clean_params', {}))
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df, **kwargs.get('impute_params', {}))
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df, encoding_method=kwargs.get('encoding_method', 'onehot'))
        
        # Scale features
        df = self.scale_features(df, method=kwargs.get('scaling_method', 'standard'))
        
        # Remove duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        
        self.processed_data = df
        logger.info("Full preprocessing pipeline complete")
        
        return df


def load_and_preprocess_data(data_path: str, **kwargs) -> Tuple[pd.DataFrame, PainManagementDataProcessor]:
    """
    Convenience function to load and preprocess data in one step.
    
    Args:
        data_path (str): Path to the Excel data file
        **kwargs: Additional arguments for preprocessing pipeline
        
    Returns:
        Tuple: (processed_dataframe, processor_instance)
    """
    processor = PainManagementDataProcessor(data_path)
    processed_data = processor.run_full_pipeline(**kwargs)
    return processed_data, processor


if __name__ == "__main__":
    # Example usage
    data_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/prehospital/analgesia/data/trauma_categories_Rega Pain Study15.09.2025_v2.xlsx'
    
    # Load and preprocess data
    processed_data, processor = load_and_preprocess_data(data_path)
    
    # Prepare for modeling
    X_train, X_test, y_train, y_test = processor.prepare_modeling_data()
    
    print("Preprocessing complete!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target distribution in training: {y_train.value_counts().to_dict()}")