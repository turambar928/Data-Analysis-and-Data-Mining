#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Diabetes Prediction Model - Complete Version
=====================================================

Focused on Dataset 2 (Lab Results) for optimal diabetes prediction.
Primary evaluation metric: F1-Score (as required)

Key Features:
- Uses HbA1c levels to create diabetes target (medical standard)
- Multiple ML algorithms optimized for F1-Score
- Handles class imbalance
- Comprehensive evaluation and new case prediction

Author: Data Analysis Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           f1_score, accuracy_score, precision_score, 
                           recall_score, roc_auc_score)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class OptimizedDiabetesPredictor:
    """
    Optimized diabetes prediction system focused on F1-Score.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_f1_score = 0
        self.feature_names = []
        self.results = {}
        
    def load_and_prepare_lab_data(self):
        """
        Load and prepare lab results data (Dataset 2) for diabetes prediction.
        """
        print("="*60)
        print("🩺 DIABETES PREDICTION - LAB DATA ANALYSIS")
        print("="*60)
        
        print("\nLoading lab results dataset...")
        try:
            df = pd.read_excel("../data/fina_project_data02.xlsx")
            print(f"✓ Dataset loaded: {df.shape}")
            
            # Key diabetes-related features
            diabetes_features = [
                '糖化血红蛋白',  # HbA1c - PRIMARY diabetes indicator
                '葡萄糖',        # Fasting glucose
                '葡萄糖(餐后2小时)',  # Post-meal glucose
                '胰岛素',        # Insulin
                'C肽1',          # C-peptide
                '胰岛素（餐后2小时）',  # Post-meal insulin
                'C肽（餐后2小时）',     # Post-meal C-peptide
                '总胆固醇',      # Total cholesterol
                '甘油三酯',      # Triglycerides
                '高密度脂蛋白',  # HDL
                '低密度脂蛋白',  # LDL
                '肌酐',          # Creatinine
                '尿素',          # Urea
                '尿酸',          # Uric acid
            ]
            
            # Select available features
            available_features = [col for col in diabetes_features if col in df.columns]
            print(f"✓ Available diabetes features: {len(available_features)}")
            
            # Create working dataset
            df_work = df[['病人姓名', '入院时间'] + available_features].copy()
            
            # Create diabetes target based on HbA1c levels
            # Medical standard: HbA1c >= 6.5% indicates diabetes
            if '糖化血红蛋白' in df_work.columns:
                df_work['diabetes_target'] = (df_work['糖化血红蛋白'] >= 6.5).astype(int)
                print(f"✓ Created diabetes target based on HbA1c >= 6.5%")
                
                # Remove rows with missing HbA1c
                df_work = df_work.dropna(subset=['糖化血红蛋白'])
                print(f"✓ Rows with valid HbA1c: {len(df_work)}")
                
                # Check target distribution
                target_dist = df_work['diabetes_target'].value_counts()
                print(f"✓ Target distribution:")
                print(f"   No Diabetes (0): {target_dist.get(0, 0)}")
                print(f"   Diabetes (1): {target_dist.get(1, 0)}")
                
                return df_work
            else:
                print("✗ HbA1c not available - cannot create reliable diabetes target")
                return None
                
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return None
    
    def preprocess_features(self, df):
        """
        Preprocess features for model training.
        """
        print(f"\n{'='*50}")
        print("FEATURE PREPROCESSING")
        print(f"{'='*50}")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['病人姓名', '入院时间', 'diabetes_target']]
        X = df[feature_cols].copy()
        y = df['diabetes_target'].copy()
        
        print(f"Original features: {len(feature_cols)}")
        print(f"Original samples: {len(X)}")
        
        # Remove features with too many missing values (>50%)
        missing_threshold = 0.5
        missing_ratios = X.isnull().sum() / len(X)
        features_to_keep = missing_ratios[missing_ratios <= missing_threshold].index
        X = X[features_to_keep]
        
        print(f"Features after removing high missing: {len(features_to_keep)}")
        
        # Fill remaining missing values with median
        for col in X.columns:
            X[col].fillna(X[col].median(), inplace=True)
        
        # Remove samples with missing target
        valid_mask = ~y.isnull()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Final target distribution: {dict(y.value_counts())}")
        
        self.feature_names = list(X.columns)
        return X, y
    
    def train_models_with_f1_focus(self, X, y):
        """
        Train multiple models optimized for F1-Score.
        """
        print(f"\n{'='*60}")
        print("MODEL TRAINING - F1-SCORE OPTIMIZATION")
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
                class_weight='balanced',
                max_depth=10,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                C=1.0,
                random_state=42,
                max_iter=1000
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                scale_pos_weight=class_counts[0]/class_counts[1],
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
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Store results
            self.results[name] = {
                'CV_F1_mean': cv_f1_scores.mean(),
                'CV_F1_std': cv_f1_scores.std(),
                'Test_F1': f1,
                'Test_Accuracy': accuracy,
                'Test_Precision': precision,
                'Test_Recall': recall,
                'Test_AUC': auc
            }
            
            print(f"CV F1-Score: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std()*2:.4f})")
            print(f"Test F1-Score: {f1:.4f} ⭐")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            if auc:
                print(f"Test AUC-ROC: {auc:.4f}")
            
            # Track best model based on F1-score
            if f1 > self.best_f1_score:
                self.best_f1_score = f1
                self.best_model = (name, model)
        
        print(f"\n{'='*50}")
        print(f"🏆 BEST MODEL: {self.best_model[0]}")
        print(f"🎯 BEST F1-SCORE: {self.best_f1_score:.4f}")
        print(f"{'='*50}")
        
        return X_test_scaled, y_test
    
    def generate_comprehensive_report(self, X_test, y_test):
        '''
        Generate comprehensive evaluation report.
        '''
        print(f"\n{'='*60}")
        print("?? COMPREHENSIVE EVALUATION REPORT")
        print(f"{'='*60}")
        
        # Results summary
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        print("\n?? Model Performance Summary (F1-Score Focus):")
        print(results_df.to_string())
        
        # Best model detailed evaluation
        best_model = self.best_model[1]
        y_pred = best_model.predict(X_test)
        
        print(f"\n?? Detailed Evaluation - {self.best_model[0]}:")
        print("-" * 50)
        print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
        
        print("\n?? Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                 Predicted")
        print(f"                No    Yes")
        print(f"Actual No    {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"Actual Yes   {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            print(f"\n?? Top 10 Most Important Features:")
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"{i+1:2d}. {row['Feature']:<20} {row['Importance']:.4f}")
        
        return results_df
    
    def predict_new_cases(self, test_cases):
        '''
        Predict diabetes for new test cases.
        '''
        if self.best_model is None:
            print("No trained model available!")
            return None
        
        print(f"\n{'='*60}")
        print("?? TESTING NEW CASES")
        print(f"{'='*60}")
        
        model = self.best_model[1]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n?? Test Case {i}:")
            
            # Create DataFrame
            case_df = pd.DataFrame([case])
            
            # Ensure all features are present
            for feature in self.feature_names:
                if feature not in case_df.columns:
                    case_df[feature] = 0  # Default value
            
            # Select and order features
            case_df = case_df[self.feature_names]
            
            # Scale features
            case_scaled = self.scaler.transform(case_df)
            
            # Make prediction
            prediction = model.predict(case_scaled)[0]
            probability = model.predict_proba(case_scaled)[0] if hasattr(model, 'predict_proba') else None
            
            print(f"   Input: {case}")
            print(f"   Prediction: {'?? DIABETES' if prediction == 1 else '?? NO DIABETES'}")
            if probability is not None:
                print(f"   Confidence: {probability[1]:.1%} diabetes risk")
        
        return True

def main():
    '''
    Main execution function.
    '''
    print("?? OPTIMIZED DIABETES PREDICTION MODEL")
    print("Primary Evaluation Metric: F1-Score")
    print("Medical Standard: HbA1c >= 6.5% = Diabetes")
    print("="*60)
    
    # Initialize predictor
    predictor = OptimizedDiabetesPredictor()
    
    # Load and prepare data
    df = predictor.load_and_prepare_lab_data()
    if df is None:
        print("? Failed to load data")
        return None
    
    # Preprocess features
    X, y = predictor.preprocess_features(df)
    if X is None:
        print("? Failed to preprocess data")
        return None
    
    # Train models
    X_test, y_test = predictor.train_models_with_f1_focus(X, y)
    
    # Generate report
    results_df = predictor.generate_comprehensive_report(X_test, y_test)
    
    # Test new cases
    test_cases = [
        {
            '糖化血红蛋白': 7.5,  # High HbA1c - likely diabetes
            '葡萄糖': 9.2,        # High glucose
            '胰岛素': 18.0,       # High insulin
        },
        {
            '糖化血红蛋白': 5.8,  # Normal HbA1c - likely no diabetes
            '葡萄糖': 5.5,        # Normal glucose
            '胰岛素': 8.0,        # Normal insulin
        },
        {
            '糖化血红蛋白': 6.7,  # Borderline HbA1c
            '葡萄糖': 7.8,        # Elevated glucose
            '胰岛素': 12.0,       # Elevated insulin
        }
    ]
    
    predictor.predict_new_cases(test_cases)
    
    print(f"\n{'='*60}")
    print("? DIABETES PREDICTION MODEL COMPLETED")
    print(f"?? Best F1-Score: {predictor.best_f1_score:.4f}")
    print(f"?? Best Model: {predictor.best_model[0]}")
    print("="*60)
    
    return predictor, results_df

if __name__ == "__main__":
    model, results = main()
