#!/usr/bin/env python3
"""
Diabetes Prediction Model
========================

This script builds and evaluates machine learning models for diabetes prediction
using health check data. The primary evaluation metric is F1-Score as required.

Author: Data Analysis Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           f1_score, accuracy_score, precision_score, 
                           recall_score, roc_auc_score, roc_curve)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictionModel:
    """
    A comprehensive diabetes prediction model with multiple algorithms
    and evaluation metrics, focusing on F1-Score optimization.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.best_f1_score = 0
        self.feature_names = []
        
    def load_data(self, file_path1, file_path2=None):
        """
        Load and combine datasets from Excel files.
        
        Args:
            file_path1 (str): Path to first dataset
            file_path2 (str): Path to second dataset (optional)
        
        Returns:
            pd.DataFrame: Combined dataset
        """
        print("Loading datasets...")
        
        # Load first dataset
        try:
            df1 = pd.read_excel(file_path1)
            print(f"Dataset 1 loaded: {df1.shape}")
            print(f"Columns: {list(df1.columns)}")
        except Exception as e:
            print(f"Error loading dataset 1: {e}")
            return None
            
        # Load second dataset if provided
        if file_path2:
            try:
                df2 = pd.read_excel(file_path2)
                print(f"Dataset 2 loaded: {df2.shape}")
                print(f"Columns: {list(df2.columns)}")
                
                # Combine datasets
                df = pd.concat([df1, df2], ignore_index=True)
                print(f"Combined dataset shape: {df.shape}")
            except Exception as e:
                print(f"Error loading dataset 2: {e}")
                df = df1
        else:
            df = df1
            
        return df
    
    def explore_data(self, df):
        """
        Perform exploratory data analysis.
        
        Args:
            df (pd.DataFrame): Dataset to explore
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nColumn Names and Types:")
        print(df.dtypes)
        
        print(f"\nMissing Values:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print(f"\nBasic Statistics:")
        print(df.describe())
        
        # Check for potential target variables
        print(f"\nPotential target variables (looking for diabetes-related columns):")
        diabetes_cols = [col for col in df.columns if 'diabetes' in col.lower() or 
                        'diabetic' in col.lower() or 'dm' in col.lower()]
        print(diabetes_cols)
        
        return df
    
    def preprocess_data(self, df, target_column):
        """
        Preprocess the data for machine learning.
        
        Args:
            df (pd.DataFrame): Raw dataset
            target_column (str): Name of target column
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"\nPreprocessing data with target: {target_column}")
        
        # Separate features and target
        if target_column not in df.columns:
            print(f"Error: Target column '{target_column}' not found!")
            return None
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values
        print("Handling missing values...")
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
            else:
                X[col].fillna(X[col].median(), inplace=True)
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Encode target variable if necessary
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.label_encoders['target'] = le_target
        
        self.feature_names = list(X.columns)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Train multiple machine learning models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Define models
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        # Train and evaluate each model using cross-validation
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation F1-scores
            cv_f1_scores = cross_val_score(model, X_train, y_train, 
                                         cv=5, scoring='f1_weighted')
            
            # Train the model
            model.fit(X_train, y_train)
            self.models[name] = model
            
            mean_f1 = cv_f1_scores.mean()
            std_f1 = cv_f1_scores.std()
            
            print(f"{name} - CV F1-Score: {mean_f1:.4f} (+/- {std_f1*2:.4f})")
            
            # Track best model based on F1-score
            if mean_f1 > self.best_f1_score:
                self.best_f1_score = mean_f1
                self.best_model = (name, model)
        
        print(f"\nBest model: {self.best_model[0]} with F1-Score: {self.best_f1_score:.4f}")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{name} Results:")
            print("-" * 30)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred, average='weighted')
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'F1-Score': f1,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall
            }
            
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_pred_proba)
                results[name]['AUC-ROC'] = auc
                print(f"AUC-ROC: {auc:.4f}")
            
            print(f"F1-Score: {f1:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix:\n{cm}")
        
        return results
    
    def predict_new_cases(self, new_data):
        """
        Predict diabetes for new cases using the best model.
        
        Args:
            new_data: New data points for prediction
            
        Returns:
            predictions and probabilities
        """
        if self.best_model is None:
            print("No trained model available!")
            return None
        
        model = self.best_model[1]
        
        # Preprocess new data (same as training data)
        new_data_scaled = self.scaler.transform(new_data)
        
        # Make predictions
        predictions = model.predict(new_data_scaled)
        probabilities = model.predict_proba(new_data_scaled) if hasattr(model, 'predict_proba') else None
        
        return predictions, probabilities

def main():
    """
    Main function to run the diabetes prediction pipeline.
    """
    # Initialize the model
    diabetes_model = DiabetesPredictionModel()
    
    # Load data
    data_path1 = "datasets/fina_project_data01.xlsx"
    data_path2 = "datasets/fina_project_data02.xlsx"
    
    df = diabetes_model.load_data(data_path1, data_path2)
    if df is None:
        return
    
    # Explore data
    df = diabetes_model.explore_data(df)
    
    # Note: You'll need to specify the correct target column name
    # This will be determined after examining the actual data
    print("\nPlease examine the data and specify the target column for diabetes prediction.")
    print("Common names might be: 'diabetes', 'diabetic', 'dm', 'target', etc.")
    
    return diabetes_model, df

if __name__ == "__main__":
    model, data = main()
