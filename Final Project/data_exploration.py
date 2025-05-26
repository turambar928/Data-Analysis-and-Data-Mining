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
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\nCategorical Columns:")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count} unique values")
                if unique_count <= 10:
                    print(f"    Values: {list(df[col].unique())}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading {dataset_name}: {e}")
        return None

def compare_datasets(df1, df2):
    """
    Compare two datasets to understand their compatibility.
    
    Args:
        df1, df2: DataFrames to compare
    """
    print(f"\n{'='*60}")
    print("DATASET COMPARISON")
    print(f"{'='*60}")
    
    if df1 is None or df2 is None:
        print("Cannot compare - one or both datasets failed to load")
        return
    
    print(f"Dataset 1 shape: {df1.shape}")
    print(f"Dataset 2 shape: {df2.shape}")
    
    # Compare columns
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1.intersection(cols2)
    unique_to_1 = cols1 - cols2
    unique_to_2 = cols2 - cols1
    
    print(f"\nColumn Comparison:")
    print(f"  Common columns: {len(common_cols)}")
    if len(common_cols) > 0:
        print(f"    {list(common_cols)}")
    
    print(f"  Unique to Dataset 1: {len(unique_to_1)}")
    if len(unique_to_1) > 0:
        print(f"    {list(unique_to_1)}")
    
    print(f"  Unique to Dataset 2: {len(unique_to_2)}")
    if len(unique_to_2) > 0:
        print(f"    {list(unique_to_2)}")
    
    # Check if datasets can be combined
    if len(common_cols) > 0:
        print(f"\n✓ Datasets can potentially be combined using common columns")
    else:
        print(f"\n⚠ Datasets have no common columns - may need separate analysis")

def create_visualizations(df, dataset_name):
    """
    Create basic visualizations for the dataset.
    
    Args:
        df: DataFrame to visualize
        dataset_name: Name for the plots
    """
    if df is None:
        return
    
    print(f"\nCreating visualizations for {dataset_name}...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Data Overview - {dataset_name}', fontsize=16)
    
    # 1. Missing values heatmap
    if df.isnull().sum().sum() > 0:
        sns.heatmap(df.isnull(), ax=axes[0,0], cbar=True, yticklabels=False)
        axes[0,0].set_title('Missing Values Pattern')
    else:
        axes[0,0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                      transform=axes[0,0].transAxes, fontsize=14)
        axes[0,0].set_title('Missing Values Pattern')
    
    # 2. Data types distribution
    dtype_counts = df.dtypes.value_counts()
    axes[0,1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[0,1].set_title('Data Types Distribution')
    
    # 3. Numeric columns correlation (if any)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title('Numeric Features Correlation')
    else:
        axes[1,0].text(0.5, 0.5, 'Insufficient Numeric Columns', ha='center', va='center',
                      transform=axes[1,0].transAxes, fontsize=12)
        axes[1,0].set_title('Numeric Features Correlation')
    
    # 4. Sample size and basic stats
    info_text = f"""
    Dataset: {dataset_name}
    Rows: {df.shape[0]:,}
    Columns: {df.shape[1]}
    Numeric: {len(df.select_dtypes(include=[np.number]).columns)}
    Categorical: {len(df.select_dtypes(include=['object']).columns)}
    Missing Values: {df.isnull().sum().sum():,}
    Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    """
    axes[1,1].text(0.1, 0.5, info_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='center')
    axes[1,1].set_title('Dataset Summary')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'Final Project/{dataset_name}_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to explore both datasets.
    """
    print("DIABETES PREDICTION - DATA EXPLORATION")
    print("=" * 60)
    
    # File paths
    file1 = "Final Project/datasets/fina_project_data01.xlsx"
    file2 = "Final Project/datasets/fina_project_data02.xlsx"
    
    # Explore both datasets
    df1 = explore_dataset(file1, "Dataset 1")
    df2 = explore_dataset(file2, "Dataset 2")
    
    # Compare datasets
    compare_datasets(df1, df2)
    
    # Create visualizations
    if df1 is not None:
        create_visualizations(df1, "Dataset_1")
    if df2 is not None:
        create_visualizations(df2, "Dataset_2")
    
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
