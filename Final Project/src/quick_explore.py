#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Quick Data Exploration for Large Excel Files
"""

import pandas as pd
import numpy as np

def quick_explore(file_path, dataset_name, nrows=1000):
    """
    Quick exploration of large dataset by loading only first N rows.
    """
    print(f"\n{'='*50}")
    print(f"QUICK EXPLORATION: {dataset_name}")
    print(f"{'='*50}")
    
    try:
        # Load only first 1000 rows for quick exploration
        df = pd.read_excel(file_path, nrows=nrows)
        print(f"鉁?Loaded first {nrows} rows from {dataset_name}")
        print(f"  Sample shape: {df.shape}")
        
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\nData Types:")
        print(df.dtypes.value_counts())
        
        print(f"\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  No missing values in sample")
        
        # Look for diabetes-related columns
        print(f"\nPotential Target Columns:")
        diabetes_keywords = ['diabetes', 'diabetic', 'dm', 'target', 'outcome', 'class', 'label']
        found_targets = []
        
        for col in df.columns:
            col_lower = col.lower()
            for keyword in diabetes_keywords:
                if keyword in col_lower:
                    found_targets.append(col)
                    unique_vals = df[col].unique()
                    print(f"  {col}: {unique_vals[:10]} ({'...' if len(unique_vals) > 10 else ''})")
                    break
        
        # Check binary columns
        print(f"\nBinary Columns (potential targets):")
        for col in df.columns:
            unique_vals = df[col].unique()
            if len(unique_vals) == 2:
                print(f"  {col}: {unique_vals}")
        
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        return df, found_targets
        
    except Exception as e:
        print(f"鉁?Error: {e}")
        return None, []

def main():
    print("QUICK DIABETES DATA EXPLORATION")
    print("=" * 50)
    
    # Explore both datasets with limited rows
    df1, targets1 = quick_explore("../data/fina_project_data01.xlsx", "Dataset 1")
    df2, targets2 = quick_explore("../data/fina_project_data02.xlsx", "Dataset 2")
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    if df1 is not None:
        print(f"Dataset 1: {df1.shape[1]} columns, potential targets: {targets1}")
    if df2 is not None:
        print(f"Dataset 2: {df2.shape[1]} columns, potential targets: {targets2}")
    
    return df1, df2

if __name__ == "__main__":
    d1, d2 = main()

