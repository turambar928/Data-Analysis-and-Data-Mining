#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Diabetes Prediction Model
======================================

This script builds a diabetes prediction model using health check data.
Primary evaluation metric: F1-Score (as required)

Key Features:
- Handles both datasets with different structures
- Creates diabetes target from HbA1c levels (Dataset 2) and conclusions (Dataset 1)
- Multiple ML algorithms with F1-Score optimization
- Comprehensive evaluation and testing

Author: Data Analysis Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           f1_score, accuracy_score, precision_score, 
                           recall_score, roc_auc_score, roc_curve)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictionSystem:
    """
    Complete diabetes prediction system with data preprocessing,
    model training, and evaluation focused on F1-Score optimization.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_f1_score = 0
        self.feature_names = []
        self.results = {}
        
    def load_and_prepare_data(self):
        """
        Load both datasets and prepare them for diabetes prediction.
        """
        print("="*60)
        print("LOADING AND PREPARING DIABETES PREDICTION DATA")
        print("="*60)
        
        # Load Dataset 2 (Lab Results) - Primary dataset for diabetes prediction
        print("\nLoading Dataset 2 (Lab Results)...")
        try:
            df2 = pd.read_excel("../data/fina_project_data02.xlsx")
            print(f"✓ Dataset 2 loaded: {df2.shape}")
            
            # Create diabetes target based on HbA1c levels
            df2_processed = self.create_diabetes_target_from_lab_data(df2)
            
        except Exception as e:
            print(f"✗ Error loading Dataset 2: {e}")
            df2_processed = None
        
        # Load Dataset 1 (Health Check) - Secondary dataset
        print("\nLoading Dataset 1 (Health Check)...")
        try:
            df1 = pd.read_excel("../data/fina_project_data01.xlsx")
            print(f"✓ Dataset 1 loaded: {df1.shape}")
            
            # Extract diabetes information from conclusions
            df1_processed = self.create_diabetes_target_from_conclusions(df1)
            
        except Exception as e:
            print(f"✗ Error loading Dataset 1: {e}")
            df1_processed = None
        
        return df2_processed, df1_processed
    
    def create_diabetes_target_from_lab_data(self, df):
        """
        Create diabetes target variable from lab results (Dataset 2).
        Uses HbA1c levels as primary indicator.
        """
        print("\nCreating diabetes target from lab data...")
        
        # Key diabetes-related columns
        diabetes_features = [
            '糖化血红蛋白',  # HbA1c - primary diabetes indicator
            '葡萄糖',        # Glucose
            '葡萄糖1', '葡萄糖2', '葡萄糖3',  # Multiple glucose measurements
            '葡萄糖(餐后2小时)',  # Post-meal glucose
            '胰岛素',        # Insulin
            'C肽1',          # C-peptide
            '胰岛素（餐后2小时）',  # Post-meal insulin
            'C肽（餐后2小时）',     # Post-meal C-peptide
        ]
        
        # Select relevant columns
        available_features = [col for col in diabetes_features if col in df.columns]
        print(f"Available diabetes features: {available_features}")
        
        # Create working dataset
        df_work = df[['病人姓名', '入院时间'] + available_features].copy()
        
        # Create diabetes target based on HbA1c levels
        # HbA1c >= 6.5% indicates diabetes
        if '糖化血红蛋白' in df_work.columns:
            df_work['diabetes_target'] = (df_work['糖化血红蛋白'] >= 6.5).astype(int)
            print(f"Created diabetes target based on HbA1c >= 6.5%")
            
            # Remove rows with missing HbA1c
            df_work = df_work.dropna(subset=['糖化血红蛋白'])
            print(f"Rows with valid HbA1c: {len(df_work)}")
            
            # Check target distribution
            target_dist = df_work['diabetes_target'].value_counts()
            print(f"Target distribution: {dict(target_dist)}")
            
        else:
            print("⚠ HbA1c not available, using glucose levels")
            # Fallback to fasting glucose >= 126 mg/dL (7.0 mmol/L)
            if '葡萄糖' in df_work.columns:
                df_work['diabetes_target'] = (df_work['葡萄糖'] >= 7.0).astype(int)
                df_work = df_work.dropna(subset=['葡萄糖'])
            else:
                print("✗ No suitable diabetes indicators found")
                return None
        
        return df_work
    
    def create_diabetes_target_from_conclusions(self, df):
        """
        Extract diabetes information from health check conclusions (Dataset 1).
        """
        print("\nExtracting diabetes info from health check conclusions...")
        
        # Look for diabetes mentions in conclusions
        df_work = df[['姓名', '年龄', '性别', '体检结论']].copy()
        
        # Create diabetes target based on conclusion text
        diabetes_keywords = ['糖尿病', '血糖', 'diabetes', '高血糖']
        
        df_work['diabetes_target'] = 0
        for keyword in diabetes_keywords:
            mask = df_work['体检结论'].str.contains(keyword, na=False, case=False)
            df_work.loc[mask, 'diabetes_target'] = 1
        
        # Remove rows with missing conclusions
        df_work = df_work.dropna(subset=['体检结论'])
        
        target_dist = df_work['diabetes_target'].value_counts()
        print(f"Extracted diabetes cases from conclusions: {dict(target_dist)}")
        
        return df_work
    
    def preprocess_features(self, df):
        """
        Preprocess features for model training.
        """
        print(f"\n{'='*50}")
        print("FEATURE PREPROCESSING")
        print(f"{'='*50}")
        
        # Separate features and target
        exclude_cols = ['病人姓名', '入院时间', '姓名', '体检结论', 'diabetes_target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['diabetes_target'].copy()
        
        print(f"Original features: {len(feature_cols)}")
        print(f"Original samples: {len(X)}")
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Remove features with too many missing values (>70%)
        missing_threshold = 0.7
        missing_ratios = X.isnull().sum() / len(X)
        features_to_keep = missing_ratios[missing_ratios <= missing_threshold].index
        X = X[features_to_keep]
        
        print(f"Features after removing high missing: {len(features_to_keep)}")
        
        # Fill remaining missing values
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0, inplace=True)
        
        # Remove samples with missing target
        valid_mask = ~y.isnull()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Final target distribution: {dict(y.value_counts())}")
        
        self.feature_names = list(X.columns)
        return X, y
    
    def train_and_evaluate_models(self, X, y):
        '''
        Train and evaluate multiple models with F1-Score focus.
        '''
        print(f"\n{'='*60}")
        print("MODEL TRAINING AND EVALUATION")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        # Calculate class weights for imbalanced data
        class_counts = y_train.value_counts()
        
        # Define models optimized for F1-Score
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced',  # Handle imbalanced data
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                random_state=42, 
                max_iter=1000
            ),
            'SVM': SVC(
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                random_state=42,
                eval_metric='logloss'
            )
        }
        
        # Train and evaluate each model
        for name, model in models_config.items():
            print(f"\n{'-'*40}")
            print(f"Training {name}...")
            
            # Cross-validation F1-scores
            cv_f1_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=5, scoring='f1'
            )
            
            # Train the model
            model.fit(X_train_scaled, y_train)
            self.models[name] = model
            
            # Test set predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'CV_F1_mean': cv_f1_scores.mean(),
                'CV_F1_std': cv_f1_scores.std(),
                'Test_F1': f1,
                'Test_Accuracy': accuracy,
                'Test_Precision': precision,
                'Test_Recall': recall
            }
            
            print(f"CV F1-Score: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std()*2:.4f})")
            print(f"Test F1-Score: {f1:.4f} ?")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            
            # Track best model based on F1-score
            if f1 > self.best_f1_score:
                self.best_f1_score = f1
                self.best_model = (name, model)
        
        print(f"\n{'='*50}")
        print(f"?? BEST MODEL: {self.best_model[0]}")
        print(f"?? BEST F1-SCORE: {self.best_f1_score:.4f}")
        print(f"{'='*50}")
        
        return X_test_scaled, y_test
    
    def create_evaluation_report(self, X_test, y_test):
        '''
        Create comprehensive evaluation report.
        '''
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        
        # Results summary table
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        print("\nModel Performance Summary:")
        print(results_df)
        
        # Best model detailed evaluation
        best_model = self.best_model[1]
        y_pred = best_model.predict(X_test)
        
        print(f"\nDetailed Evaluation - {self.best_model[0]}:")
        print("-" * 40)
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            print(f"\nTop 10 Most Important Features:")
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"{i+1:2d}. {row['Feature']:<20} {row['Importance']:.4f}")
        
        return results_df
    
    def predict_new_cases(self, new_data_dict):
        '''
        Predict diabetes for new cases.
        
        Args:
            new_data_dict: Dictionary with feature values for new cases
        '''
        if self.best_model is None:
            print("No trained model available!")
            return None
        
        print(f"\n{'='*40}")
        print("PREDICTING NEW CASES")
        print(f"{'='*40}")
        
        # Create DataFrame from input
        new_df = pd.DataFrame([new_data_dict])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in new_df.columns:
                new_df[feature] = 0  # Default value
        
        # Select and order features
        new_df = new_df[self.feature_names]
        
        # Scale features
        new_data_scaled = self.scaler.transform(new_df)
        
        # Make prediction
        model = self.best_model[1]
        prediction = model.predict(new_data_scaled)[0]
        probability = model.predict_proba(new_data_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        print(f"Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
        if probability is not None:
            print(f"Probability: {probability[1]:.4f} (diabetes)")
        
        return prediction, probability

def main():
    '''
    Main execution function.
    '''
    print("?? DIABETES PREDICTION MODEL")
    print("Primary Evaluation Metric: F1-Score")
    print("="*60)
    
    # Initialize system
    diabetes_system = DiabetesPredictionSystem()
    
    # Load and prepare data
    df2, df1 = diabetes_system.load_and_prepare_data()
    
    # Use Dataset 2 (lab results) as primary data source
    if df2 is not None and len(df2) > 100:
        print(f"\n? Using Dataset 2 (Lab Results) for model training")
        working_df = df2
    elif df1 is not None and len(df1) > 100:
        print(f"\n? Using Dataset 1 (Health Check) for model training")
        working_df = df1
    else:
        print("? Insufficient data for model training")
        return None
    
    # Preprocess features
    X, y = diabetes_system.preprocess_features(working_df)
    if X is None:
        return None
    
    # Train and evaluate models
    X_test, y_test = diabetes_system.train_and_evaluate_models(X, y)
    
    # Create evaluation report
    results_df = diabetes_system.create_evaluation_report(X_test, y_test)
    
    # Example prediction for new case
    print(f"\n{'='*60}")
    print("EXAMPLE: PREDICTING NEW CASE")
    print(f"{'='*60}")
    
    # Example new case (adjust based on available features)
    example_case = {
        '糖化血红蛋白': 7.2,  # HbA1c = 7.2% (indicates diabetes)
        '葡萄糖': 8.5,        # Glucose = 8.5 mmol/L
        '胰岛素': 15.0,       # Insulin level
    }
    
    diabetes_system.predict_new_cases(example_case)
    
    return diabetes_system, results_df

if __name__ == "__main__":
    model_system, results = main()
