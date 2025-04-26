import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load the dataset
file_path = "anonymized data for workshop.xlsx"
print(f"Loading data from: {file_path}")
df = pd.read_excel(file_path)

# Basic information about the dataset
print("\n=== DATASET OVERVIEW ===")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\n=== COLUMN INFORMATION ===")
print(df.info())

print("\n=== SUMMARY STATISTICS ===")
print(df.describe().T)

print("\n=== MISSING VALUES ===")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
print(missing_df[missing_df['Missing Values'] > 0])

# Save the initial analysis to a file
with open('initial_analysis_results.txt', 'w') as f:
    f.write(f"Dataset shape: {df.shape}\n\n")
    f.write("Column Data Types:\n")
    f.write(str(df.dtypes) + "\n\n")
    f.write("Summary Statistics:\n")
    f.write(str(df.describe().T) + "\n\n")
    f.write("Missing Values:\n")
    f.write(str(missing_df[missing_df['Missing Values'] > 0]))

print("\nInitial analysis completed and saved to 'initial_analysis_results.txt'")
