#!/usr/bin/env python3
"""
Data Exploration Script for Diabetes Prediction
===============================================

This script explores the diabetes datasets to understand their structure,
identify the target variable, and prepare for model development.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def explore_dataset(file_path, dataset_name):
    """
    Explore a single dataset and return detailed information.
    
    Args:
        file_path (str): Path to the Excel file
        dataset_name (str): Name identifier for the dataset
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    print(f"\n{'='*60}")
    print(f"EXPLORING {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Load the dataset
        df = pd.read_excel(file_path)
        print(f"✓ Successfully loaded {dataset_name}")
        print(f"  Shape: {df.shape}")
        
        # Basic information
        print(f"\nColumn Information:")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Column names: {list(df.columns)}")
        
        # Data types
        print(f"\nData Types:")
        for dtype in df.dtypes.value_counts().index:
            count = df.dtypes.value_counts()[dtype]
            print(f"  {dtype}: {count} columns")
        
        # Missing values
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        print(f"\nMissing Values:")
        if len(missing_cols) > 0:
            for col, missing_count in missing_cols.items():
                percentage = (missing_count / len(df)) * 100
                print(f"  {col}: {missing_count} ({percentage:.2f}%)")
        else:
            print("  No missing values found!")
        
        # Look for potential target variables
        print(f"\nPotential Target Variables:")
        diabetes_keywords = ['diabetes', 'diabetic', 'dm', 'target', 'outcome', 'class', 'label']
        potential_targets = []
        
        for col in df.columns:
            col_lower = col.lower()
            for keyword in diabetes_keywords:
                if keyword in col_lower:
                    potential_targets.append(col)
                    unique_vals = df[col].unique()
                    print(f"  {col}: {unique_vals} (count: {len(unique_vals)})")
                    break
        
        if not potential_targets:
            print("  No obvious target variables found. Checking binary columns...")
            for col in df.columns:
                unique_vals = df[col].unique()
                if len(unique_vals) == 2:
                    print(f"  Binary column - {col}: {unique_vals}")
        
        # Statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric Columns Summary:")
            print(df[numeric_cols].describe())
        
        # Show first few rows
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading {dataset_name}: {e}")
        return None

def main():
    """
    Main function to explore both datasets.
    """
    print("DIABETES PREDICTION - DATA EXPLORATION")
    print("=" * 60)
    
    # File paths (corrected for running from Final Project directory)
    file1 = "datasets/fina_project_data01.xlsx"
    file2 = "datasets/fina_project_data02.xlsx"
    
    # Explore both datasets
    df1 = explore_dataset(file1, "Dataset 1")
    df2 = explore_dataset(file2, "Dataset 2")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if df1 is not None and df2 is not None:
        print("✓ Both datasets loaded successfully")
        print("✓ Proceed with data preprocessing and model development")
        print("✓ Consider combining datasets if they have compatible structure")
        print("✓ Identify the correct target variable for diabetes prediction")
    elif df1 is not None or df2 is not None:
        print("⚠ Only one dataset loaded successfully")
        print("✓ Proceed with available dataset")
    else:
        print("✗ Failed to load datasets - check file paths and formats")
    
    return df1, df2

if __name__ == "__main__":
    dataset1, dataset2 = main()
