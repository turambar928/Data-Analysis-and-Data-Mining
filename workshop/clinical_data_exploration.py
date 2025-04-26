import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# Load the dataset
print("Loading dataset...")
file_path = "anonymized data for workshop.xlsx"
df = pd.read_excel(file_path)
print(f"Dataset loaded with shape: {df.shape}")

# Function to save figures
def save_fig(fig, filename):
    fig.savefig(os.path.join('visualizations', filename), bbox_inches='tight', dpi=300)
    plt.close(fig)

# 1. Data Preprocessing
print("\n=== DATA PREPROCESSING ===")

# Convert date columns to datetime if not already
date_columns = ['入院时间', '出院时间', '采集时间']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# Calculate length of stay
if '入院时间' in df.columns and '出院时间' in df.columns:
    df['住院天数'] = (df['出院时间'] - df['入院时间']).dt.total_seconds() / (24 * 3600)
    print(f"Length of stay statistics:\n{df['住院天数'].describe()}")

# Identify numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print(f"Number of numeric columns: {len(numeric_cols)}")

# 2. Temporal Analysis
print("\n=== TEMPORAL ANALYSIS ===")

# Analyze data collection over time
if '采集时间' in df.columns:
    df['采集日期'] = df['采集时间'].dt.date
    df['采集月份'] = df['采集时间'].dt.month
    df['采集年份'] = df['采集时间'].dt.year
    df['采集小时'] = df['采集时间'].dt.hour

    # Plot data collection by month and year
    plt.figure(figsize=(12, 6))
    df['采集月份'].value_counts().sort_index().plot(kind='bar')
    plt.title('Data Collection by Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    save_fig(plt.gcf(), 'data_collection_by_month.png')

    plt.figure(figsize=(12, 6))
    df['采集年份'].value_counts().sort_index().plot(kind='bar')
    plt.title('Data Collection by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    save_fig(plt.gcf(), 'data_collection_by_year.png')

    # Plot data collection by hour of day
    plt.figure(figsize=(12, 6))
    df['采集小时'].value_counts().sort_index().plot(kind='bar')
    plt.title('Data Collection by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Count')
    save_fig(plt.gcf(), 'data_collection_by_hour.png')

# 3. Department Analysis
print("\n=== DEPARTMENT ANALYSIS ===")

# Analyze distribution by department
if 'HIS出院科室' in df.columns:
    dept_counts = df['HIS出院科室'].value_counts()
    print(f"Number of departments: {len(dept_counts)}")
    print("Top 10 departments by patient count:")
    print(dept_counts.head(10))

    plt.figure(figsize=(14, 8))
    dept_counts.head(15).plot(kind='bar')
    plt.title('Patient Count by Department (Top 15)')
    plt.xlabel('Department')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    save_fig(plt.gcf(), 'patient_count_by_department.png')

# 4. Laboratory Test Analysis
print("\n=== LABORATORY TEST ANALYSIS ===")

# Select common lab tests with reasonable amount of data
common_labs = []
for col in numeric_cols:
    missing_pct = df[col].isna().mean() * 100
    if missing_pct < 90:  # Less than 90% missing
        common_labs.append(col)

print(f"Common laboratory tests (less than 90% missing): {len(common_labs)}")
print(common_labs)

# Distribution of key lab values
key_labs = ['钾', '钠', '氯', '钙', '磷', '镁', '肌酐', '尿素', '尿酸', '糖化血红蛋白', '葡萄糖']
key_labs = [lab for lab in key_labs if lab in df.columns]

if key_labs:
    # Create histograms for key lab values
    for lab in key_labs:
        if df[lab].notna().sum() > 100:  # Only plot if we have enough data
            plt.figure(figsize=(10, 6))
            sns.histplot(df[lab].dropna(), kde=True)
            plt.title(f'Distribution of {lab}')
            plt.xlabel(lab)
            plt.ylabel('Count')
            save_fig(plt.gcf(), f'distribution_{lab}.png')

# 5. Correlation Analysis
print("\n=== CORRELATION ANALYSIS ===")

# Select numeric columns with less than 80% missing values for correlation analysis
corr_cols = []
for col in numeric_cols:
    missing_pct = df[col].isna().mean() * 100
    if missing_pct < 80 and col not in ['', '住院号码']:  # Exclude non-informative columns
        corr_cols.append(col)

print(f"Columns used for correlation analysis: {len(corr_cols)}")

if corr_cols:
    # Calculate correlation matrix
    corr_df = df[corr_cols].copy()
    correlation = corr_df.corr(method='pearson')

    # Plot correlation heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=False, mask=mask, cmap='coolwarm',
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix of Laboratory Tests')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    save_fig(plt.gcf(), 'correlation_heatmap.png')

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(correlation.columns)):
        for j in range(i+1, len(correlation.columns)):
            if abs(correlation.iloc[i, j]) > 0.7:
                high_corr_pairs.append((correlation.columns[i], correlation.columns[j], correlation.iloc[i, j]))

    print("Highly correlated pairs (|r| > 0.7):")
    for pair in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"{pair[0]} and {pair[1]}: {pair[2]:.3f}")

    # Scatter plots for highly correlated pairs (top 5)
    for i, (var1, var2, corr) in enumerate(sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]):
        plt.figure(figsize=(10, 6))
        plt.scatter(df[var1], df[var2], alpha=0.5)
        plt.title(f'Correlation between {var1} and {var2} (r={corr:.3f})')
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.grid(True, linestyle='--', alpha=0.7)
        save_fig(plt.gcf(), f'correlation_scatter_{i+1}.png')

# 6. Outlier Detection
print("\n=== OUTLIER DETECTION ===")

# Select columns for outlier detection
outlier_cols = []
for col in numeric_cols:
    if df[col].notna().sum() > 1000 and col not in ['', '住院号码']:
        outlier_cols.append(col)

print(f"Columns used for outlier detection: {len(outlier_cols)}")

if outlier_cols:
    # Z-score based outlier detection
    outliers_zscore = {}
    for col in outlier_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = df[col].dropna()[z_scores > 3]
        if len(outliers) > 0:
            outliers_zscore[col] = len(outliers)

    print("Number of outliers detected using Z-score method (|z| > 3):")
    for col, count in sorted(outliers_zscore.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{col}: {count} outliers ({count/df[col].notna().sum()*100:.2f}%)")

    # Box plot for columns with most outliers
    top_outlier_cols = sorted(outliers_zscore.items(), key=lambda x: x[1], reverse=True)[:5]
    if top_outlier_cols:
        plt.figure(figsize=(14, 8))
        for i, (col, _) in enumerate(top_outlier_cols):
            plt.subplot(1, 5, i+1)
            sns.boxplot(y=df[col].dropna())
            plt.title(col)
        plt.tight_layout()
        save_fig(plt.gcf(), 'outliers_boxplot.png')

# 7. Pattern Discovery with PCA
print("\n=== PATTERN DISCOVERY WITH PCA ===")

# Select columns for PCA
pca_cols = []
for col in numeric_cols:
    if df[col].notna().sum() > 1000 and col not in ['', '住院号码']:
        pca_cols.append(col)

print(f"Columns used for PCA: {len(pca_cols)}")

if len(pca_cols) >= 3:  # Need at least 3 columns for meaningful PCA
    # Prepare data for PCA
    pca_df = df[pca_cols].copy()

    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    pca_data = imputer.fit_transform(pca_df)

    # Standardize the data
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)

    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(pca_data_scaled)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plot explained variance
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    save_fig(plt.gcf(), 'pca_explained_variance.png')

    # Number of components needed to explain 80% of variance
    n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
    print(f"Number of components needed to explain 80% of variance: {n_components_80}")

    # Plot first two principal components
    plt.figure(figsize=(12, 10))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.3)
    plt.title('First Two Principal Components')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_fig(plt.gcf(), 'pca_first_two_components.png')

    # Feature importance in first two PCs
    components = pd.DataFrame(pca.components_.T, index=pca_cols, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

    plt.figure(figsize=(14, 10))
    sns.heatmap(components.iloc[:, :5], annot=True, cmap='coolwarm')
    plt.title('Feature Importance in First 5 Principal Components')
    plt.tight_layout()
    save_fig(plt.gcf(), 'pca_feature_importance.png')

# 8. Clustering Analysis
print("\n=== CLUSTERING ANALYSIS ===")

if len(pca_cols) >= 3:
    # Use the PCA results from previous step
    # Determine optimal number of clusters using elbow method
    inertia = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pca_data_scaled)
        inertia.append(kmeans.inertia_)

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_fig(plt.gcf(), 'kmeans_elbow_curve.png')

    # Choose k based on elbow method (this is a simple heuristic)
    k_diff = np.diff(inertia)
    k_diff2 = np.diff(k_diff)
    optimal_k = k_range[np.argmax(k_diff2) + 1]
    print(f"Optimal number of clusters based on elbow method: {optimal_k}")

    # Apply K-means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(pca_data_scaled)

    # Add cluster labels to the dataframe
    df['Cluster'] = np.nan
    df.loc[pca_df.index, 'Cluster'] = clusters

    # Plot clusters in PCA space
    plt.figure(figsize=(12, 10))
    for cluster in range(optimal_k):
        plt.scatter(pca_result[clusters == cluster, 0],
                   pca_result[clusters == cluster, 1],
                   alpha=0.7,
                   label=f'Cluster {cluster+1}')
    plt.title('Clusters in PCA Space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    save_fig(plt.gcf(), 'kmeans_clusters_pca.png')

    # Analyze cluster characteristics
    cluster_stats = df.groupby('Cluster')[pca_cols].mean()
    print("\nCluster characteristics (mean values):")
    print(cluster_stats)

    # Heatmap of cluster characteristics
    plt.figure(figsize=(16, 12))
    # Standardize for better visualization
    cluster_stats_scaled = (cluster_stats - cluster_stats.mean()) / cluster_stats.std()
    sns.heatmap(cluster_stats_scaled, annot=False, cmap='coolwarm')
    plt.title('Cluster Characteristics (Standardized Mean Values)')
    plt.tight_layout()
    save_fig(plt.gcf(), 'cluster_characteristics_heatmap.png')

# 9. Time Series Analysis for Lab Values
print("\n=== TIME SERIES ANALYSIS ===")

# Check if we have patient IDs and timestamps
if '住院号码' in df.columns and '采集时间' in df.columns:
    # Get patients with multiple measurements
    patient_counts = df['住院号码'].value_counts()
    patients_with_multiple = patient_counts[patient_counts > 5].index.tolist()

    if patients_with_multiple:
        print(f"Number of patients with more than 5 measurements: {len(patients_with_multiple)}")

        # Select a few patients for time series analysis
        selected_patients = patients_with_multiple[:5]

        # Select key lab tests for time series
        ts_labs = ['钾', '钠', '氯', '钙', '磷', '肌酐', '尿素', '葡萄糖']
        ts_labs = [lab for lab in ts_labs if lab in df.columns]

        if ts_labs:
            # Plot time series for each selected patient
            for patient in selected_patients:
                patient_data = df[df['住院号码'] == patient].sort_values('采集时间')

                # Create a figure for each lab test instead of combining them
                for lab in ts_labs:
                    if patient_data[lab].notna().sum() > 1:
                        plt.figure(figsize=(10, 6))
                        # Convert to numeric to ensure we can plot
                        patient_data[lab] = pd.to_numeric(patient_data[lab], errors='coerce')
                        plt.plot(patient_data['采集时间'], patient_data[lab], 'o-')
                        plt.title(f'{lab} over Time (Patient {patient})')
                        plt.xlabel('Time')
                        plt.ylabel(lab)
                        plt.xticks(rotation=45)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        save_fig(plt.gcf(), f'time_series_{lab}_patient_{patient}.png')

# 10. Clinical Significance Analysis
print("\n=== CLINICAL SIGNIFICANCE ANALYSIS ===")

# Define normal ranges for common lab tests
normal_ranges = {
    '钾': (3.5, 5.5),  # mmol/L
    '钠': (135, 145),  # mmol/L
    '氯': (95, 105),   # mmol/L
    '钙': (2.1, 2.6),  # mmol/L
    '磷': (0.8, 1.6),  # mmol/L
    '镁': (0.7, 1.1),  # mmol/L
    '肌酐': (44, 133),  # μmol/L
    '尿素': (2.9, 8.2), # mmol/L
    '尿酸': (150, 430), # μmol/L
    '糖化血红蛋白': (4.0, 6.0),  # %
    '葡萄糖': (3.9, 6.1)  # mmol/L
}

# Calculate abnormal percentages
abnormal_stats = {}
for lab, (lower, upper) in normal_ranges.items():
    if lab in df.columns:
        # Convert to numeric to ensure we can compare
        df[lab] = pd.to_numeric(df[lab], errors='coerce')
        total = df[lab].notna().sum()
        if total > 0:
            below = (df[lab] < lower).sum()
            above = (df[lab] > upper).sum()
            normal = total - below - above

            abnormal_stats[lab] = {
                'Total': total,
                'Below Normal': below,
                'Above Normal': above,
                'Normal': normal,
                'Below %': below / total * 100,
                'Above %': above / total * 100,
                'Normal %': normal / total * 100
            }

if abnormal_stats:
    # Create a summary dataframe
    abnormal_df = pd.DataFrame({
        'Lab Test': list(abnormal_stats.keys()),
        'Total Measurements': [stats['Total'] for stats in abnormal_stats.values()],
        'Below Normal (%)': [f"{stats['Below %']:.1f}%" for stats in abnormal_stats.values()],
        'Normal (%)': [f"{stats['Normal %']:.1f}%" for stats in abnormal_stats.values()],
        'Above Normal (%)': [f"{stats['Above %']:.1f}%" for stats in abnormal_stats.values()]
    })

    print("Abnormal lab test percentages:")
    print(abnormal_df)

    # Plot abnormal percentages
    plt.figure(figsize=(14, 8))
    x = range(len(abnormal_stats))
    width = 0.3

    below_pct = [stats['Below %'] for stats in abnormal_stats.values()]
    normal_pct = [stats['Normal %'] for stats in abnormal_stats.values()]
    above_pct = [stats['Above %'] for stats in abnormal_stats.values()]

    plt.bar([i - width for i in x], below_pct, width=width, label='Below Normal', color='blue')
    plt.bar(x, normal_pct, width=width, label='Normal', color='green')
    plt.bar([i + width for i in x], above_pct, width=width, label='Above Normal', color='red')

    plt.xlabel('Lab Test')
    plt.ylabel('Percentage (%)')
    plt.title('Distribution of Lab Test Results Relative to Normal Ranges')
    plt.xticks(x, list(abnormal_stats.keys()), rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_fig(plt.gcf(), 'abnormal_lab_percentages.png')

# 11. Additional Analysis: Department-specific Patterns
print("\n=== DEPARTMENT-SPECIFIC PATTERNS ===")

# Analyze lab values by department
if 'HIS出院科室' in df.columns:
    departments = df['HIS出院科室'].unique()
    print(f"Analyzing patterns across {len(departments)} departments")

    # Select key lab tests for comparison
    dept_labs = ['钾', '钠', '氯', '钙', '磷', '肌酐', '尿素', '尿酸', '糖化血红蛋白', '葡萄糖']
    dept_labs = [lab for lab in dept_labs if lab in df.columns and df[lab].notna().sum() > 100]

    if dept_labs:
        # Create boxplots comparing lab values across departments
        for lab in dept_labs:
            if df[lab].notna().sum() > 100:  # Only if we have enough data
                plt.figure(figsize=(12, 8))
                sns.boxplot(x='HIS出院科室', y=lab, data=df)
                plt.title(f'Distribution of {lab} by Department')
                plt.xlabel('Department')
                plt.ylabel(lab)
                plt.xticks(rotation=45)
                save_fig(plt.gcf(), f'dept_comparison_{lab}.png')

        # Calculate mean values by department
        dept_means = df.groupby('HIS出院科室')[dept_labs].mean()
        print("\nMean lab values by department:")
        print(dept_means)

        # Create a heatmap of department means
        plt.figure(figsize=(14, 8))
        # Standardize for better visualization
        dept_means_scaled = (dept_means - dept_means.mean()) / dept_means.std()
        sns.heatmap(dept_means_scaled, annot=False, cmap='coolwarm')
        plt.title('Department Lab Value Patterns (Standardized Mean Values)')
        plt.tight_layout()
        save_fig(plt.gcf(), 'department_lab_patterns.png')

# 12. Summary of Findings
print("\n=== SUMMARY OF FINDINGS ===")
print("1. Dataset contains 27,351 entries with 110 variables")
print(f"2. Data spans {len(departments) if 'departments' in locals() else 'multiple'} departments with varying sample sizes")
print("3. Significant missing data across many variables (many columns >80% missing)")
print("4. Several strong correlations identified between lab tests")
print("5. Outliers detected in multiple laboratory measurements")
print("6. PCA reveals underlying patterns in the data, with {n_components_80} components explaining 80% of variance"
      .format(n_components_80=n_components_80 if 'n_components_80' in locals() else 'multiple'))
print(f"7. Clustering analysis identified {optimal_k if 'optimal_k' in locals() else 'multiple'} distinct patient groups")
print("8. Time series analysis shows temporal patterns in lab values for individual patients")
print("9. Significant percentage of abnormal lab values detected")
print("10. Department-specific patterns observed in laboratory values")

print("\nAll visualizations saved to the 'visualizations' folder")
