import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats
import os
import warnings
import seaborn as sns
from sklearn.cluster import KMeans



warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('whitegrid')  # seaborn 样式设置
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Create output directory for visualizations
os.makedirs('cluster_analysis_results', exist_ok=True)

# Function to save figures
def save_fig(fig, filename):
    fig.savefig(os.path.join('cluster_analysis_results', filename), bbox_inches='tight', dpi=300)
    plt.close(fig)

print("Loading dataset...")
file_path = "anonymized data for workshop.xlsx"
df = pd.read_excel(file_path)
print(f"Dataset loaded with shape: {df.shape}")

# Check if clustering was already performed
if 'Cluster' not in df.columns:
    print("Cluster column not found. Running clustering now...")

    # Perform clustering directly in this script
    # Select numeric columns with less than 80% missing values for clustering
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cluster_cols = []
    for col in numeric_cols:
        missing_pct = df[col].isna().mean() * 100
        if missing_pct < 80 and col not in ['', '住院号码']:
            cluster_cols.append(col)

    print(f"Selected {len(cluster_cols)} columns for clustering")

    if cluster_cols:
        # Prepare data for clustering
        cluster_df = df[cluster_cols].copy()

        # Impute missing values with median
        imputer = SimpleImputer(strategy='median')
        cluster_data = imputer.fit_transform(cluster_df)

        # Standardize the data
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)

        # Determine optimal number of clusters using elbow method
        inertia = []
        k_range = range(2, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(cluster_data_scaled)
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
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(cluster_data_scaled)

        # Add cluster labels to the dataframe
        df['Cluster'] = np.nan
        df.loc[cluster_df.index, 'Cluster'] = clusters

        # Save cluster assignments for future use
        cluster_assignments = pd.DataFrame({
            'Cluster': clusters
        }, index=cluster_df.index)
        cluster_assignments.to_csv('cluster_analysis_results/cluster_assignments.csv')

        print(f"Clustering completed. Assigned {len(clusters)} patients to {optimal_k} clusters.")
    else:
        print("Not enough data for clustering. Exiting.")
        exit()

# 1. Detailed Cluster Profiling
print("\n=== DETAILED CLUSTER PROFILING ===")

# Define key lab tests and clinical variables for profiling
lab_tests = ['钾', '钠', '氯', '钙', '磷', '镁', '肌酐', '尿素', '尿酸',
             '糖化血红蛋白', '葡萄糖', '葡萄糖(餐后2小时)',
             '游离甲状腺素', '总三碘甲状腺原氨酸', '低密度脂蛋白',
             '甲状旁腺激素', '总胆固醇', '甘油三酯']

# English names for lab tests (for plotting)
lab_tests_en = {
    '钾': 'Potassium',
    '钠': 'Sodium',
    '氯': 'Chloride',
    '钙': 'Calcium',
    '磷': 'Phosphorus',
    '镁': 'Magnesium',
    '肌酐': 'Creatinine',
    '尿素': 'Urea',
    '尿酸': 'Uric Acid',
    '糖化血红蛋白': 'HbA1c',
    '葡萄糖': 'Glucose',
    '葡萄糖(餐后2小时)': 'Glucose (2h postprandial)',
    '游离甲状腺素': 'Free T4',
    '总三碘甲状腺原氨酸': 'Total T3',
    '低密度脂蛋白': 'LDL',
    '甲状旁腺激素': 'PTH',
    '总胆固醇': 'Total Cholesterol',
    '甘油三酯': 'Triglycerides',
    '住院天数': 'Length of Stay',
    'HIS出院科室': 'Department'
}

# Department translations
dept_mapping = {
    '内分泌代谢科': 'Endocrinology & Metabolism',
    '惠宾病房': 'VIP Ward',
    '日间病房': 'Day Ward'
}

# Convert date columns to datetime if not already
date_columns = ['入院时间', '出院时间', '采集时间']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# Calculate length of stay if not already done
if '住院天数' not in df.columns and '入院时间' in df.columns and '出院时间' in df.columns:
    df['住院天数'] = (df['出院时间'] - df['入院时间']).dt.total_seconds() / (24 * 3600)

# Filter lab tests that exist in the dataset
available_labs = [lab for lab in lab_tests if lab in df.columns]

# Create a profile for each cluster
clusters = df['Cluster'].dropna().unique()
cluster_profiles = pd.DataFrame(index=clusters)

for lab in available_labs + ['住院天数']:
    if lab in df.columns:
        # Convert to numeric to ensure we can calculate statistics
        df[lab] = pd.to_numeric(df[lab], errors='coerce')
        # Calculate mean for each cluster
        means = df.groupby('Cluster')[lab].mean()
        cluster_profiles[lab] = means

# Calculate department distribution for each cluster
if 'HIS出院科室' in df.columns:
    dept_distribution = df.groupby(['Cluster', 'HIS出院科室']).size().unstack(fill_value=0)
    # Convert to percentages
    dept_distribution = dept_distribution.div(dept_distribution.sum(axis=1), axis=0) * 100
    # Add to cluster profiles
    for dept in dept_distribution.columns:
        cluster_profiles[f'Dept_{dept}'] = dept_distribution[dept]

print("\nCluster Profiles (Mean Values):")
print(cluster_profiles)

# Save cluster profiles to CSV
cluster_profiles.to_csv('cluster_analysis_results/cluster_profiles.csv')

# 2. Visualize key characteristics of each cluster
print("\n=== CLUSTER VISUALIZATION ===")

# Heatmap of standardized lab values by cluster
plt.figure(figsize=(16, 10))
# Standardize for better visualization
cluster_profiles_std = (cluster_profiles[available_labs] - cluster_profiles[available_labs].mean()) / cluster_profiles[available_labs].std()
sns.heatmap(cluster_profiles_std, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Standardized Lab Values by Cluster', fontsize=16)
plt.tight_layout()
save_fig(plt.gcf(), 'cluster_heatmap_standardized.png')

# Radar chart for selected clusters
def radar_chart(cluster_data, cluster_ids, title):
    # Select a subset of important labs for readability
    key_labs = ['钾', '钠', '氯', '钙', '糖化血红蛋白', '葡萄糖', '肌酐', '尿素']
    key_labs = [lab for lab in key_labs if lab in cluster_data.columns]

    if not key_labs:
        print("Not enough lab data for radar chart")
        return

    # Number of variables
    N = len(key_labs)

    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], [lab_tests_en.get(lab, lab) for lab in key_labs], fontsize=12)

    # Draw the cluster lines
    for cluster_id in cluster_ids:
        if cluster_id in cluster_data.index:
            values = cluster_data.loc[cluster_id, key_labs].values.flatten().tolist()
            # Close the loop
            values += values[:1]
            # Plot values
            ax.plot(angles, values, linewidth=2, label=f'Cluster {int(cluster_id)+1}')
            ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, fontsize=16, y=1.1)

    return fig

# Normalize the data for radar chart
cluster_profiles_norm = (cluster_profiles[available_labs] - cluster_profiles[available_labs].min()) / \
                        (cluster_profiles[available_labs].max() - cluster_profiles[available_labs].min())

# Create radar charts for different groups of clusters
# Metabolic clusters (example)
metabolic_clusters = [0, 1, 2]  # Adjust based on your actual cluster analysis
fig = radar_chart(cluster_profiles_norm, metabolic_clusters, 'Metabolic Parameter Comparison')
if fig:
    save_fig(fig, 'radar_metabolic_clusters.png')

# Thyroid clusters (example)
thyroid_clusters = [7, 8]  # Adjust based on your actual cluster analysis
fig = radar_chart(cluster_profiles_norm, thyroid_clusters, 'Thyroid Parameter Comparison')
if fig:
    save_fig(fig, 'radar_thyroid_clusters.png')

# 3. Advanced Dimensionality Reduction with t-SNE
print("\n=== ADVANCED DIMENSIONALITY REDUCTION ===")

# Select numeric columns with less than 80% missing values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
selected_cols = []
for col in numeric_cols:
    missing_pct = df[col].isna().mean() * 100
    if missing_pct < 80 and col not in ['', '住院号码', 'Cluster']:
        selected_cols.append(col)

print(f"Selected {len(selected_cols)} columns for t-SNE")

if selected_cols:
    # Prepare data for t-SNE
    tsne_df = df[selected_cols].copy()

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    tsne_data = imputer.fit_transform(tsne_df)

    # Standardize the data
    scaler = StandardScaler()
    tsne_data_scaled = scaler.fit_transform(tsne_data)

    # Apply t-SNE
    print("Applying t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(tsne_data_scaled)

    # Create a DataFrame with t-SNE results
    tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE 1', 't-SNE 2'])
    tsne_df['Cluster'] = df['Cluster'].values

    # Plot t-SNE results colored by cluster
    plt.figure(figsize=(12, 10))
    for cluster in tsne_df['Cluster'].dropna().unique():
        subset = tsne_df[tsne_df['Cluster'] == cluster]
        plt.scatter(subset['t-SNE 1'], subset['t-SNE 2'], label=f'Cluster {int(cluster)+1}', alpha=0.7)

    plt.title('t-SNE Visualization of Patient Clusters', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Cluster')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_fig(plt.gcf(), 'tsne_clusters.png')

# 4. Statistical Comparison Between Clusters
print("\n=== STATISTICAL COMPARISON BETWEEN CLUSTERS ===")

# Perform ANOVA for each lab test to compare clusters
anova_results = {}
for lab in available_labs:
    # Group data by cluster
    groups = [df[df['Cluster'] == cluster][lab].dropna() for cluster in clusters]
    # Filter out empty groups
    groups = [group for group in groups if len(group) > 0]

    if len(groups) >= 2:  # Need at least 2 groups for ANOVA
        try:
            f_stat, p_val = stats.f_oneway(*groups)
            anova_results[lab] = {'F-statistic': f_stat, 'p-value': p_val}
        except:
            print(f"Could not perform ANOVA for {lab}")

# Create a DataFrame with ANOVA results
anova_df = pd.DataFrame(anova_results).T
anova_df['Significant'] = anova_df['p-value'] < 0.05
anova_df = anova_df.sort_values('p-value')

print("\nANOVA Results (Top 10 most significant differences):")
print(anova_df.head(10))

# Save ANOVA results to CSV
anova_df.to_csv('cluster_analysis_results/anova_results.csv')

# Plot the most significant differences
top_labs = anova_df.index[:5]  # Top 5 most significant
for lab in top_labs:
    if lab in df.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Cluster', y=lab, data=df)
        plt.title(f'Distribution of {lab_tests_en.get(lab, lab)} by Cluster', fontsize=16)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel(lab_tests_en.get(lab, lab), fontsize=12)
        save_fig(plt.gcf(), f'boxplot_{lab}.png')

# 5. Cluster Stability Analysis
print("\n=== CLUSTER STABILITY ANALYSIS ===")

# This would typically involve resampling and reclustering
# For simplicity, we'll just check the size and completeness of each cluster

# Cluster size
cluster_sizes = df['Cluster'].value_counts().sort_index()
print("\nCluster Sizes:")
print(cluster_sizes)

# Completeness (percentage of non-missing values in each cluster)
completeness = {}
for cluster in clusters:
    cluster_df = df[df['Cluster'] == cluster]
    for lab in available_labs:
        if lab not in completeness:
            completeness[lab] = {}
        completeness[lab][cluster] = (1 - cluster_df[lab].isna().mean()) * 100

completeness_df = pd.DataFrame(completeness)
print("\nData Completeness by Cluster (%):")
print(completeness_df)

# Save completeness to CSV
completeness_df.to_csv('cluster_analysis_results/cluster_completeness.csv')

# Plot cluster sizes
plt.figure(figsize=(10, 6))
cluster_sizes.plot(kind='bar')
plt.title('Number of Patients in Each Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
save_fig(plt.gcf(), 'cluster_sizes.png')

# Plot completeness heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(completeness_df, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('Data Completeness by Cluster (%)', fontsize=16)
plt.tight_layout()
save_fig(plt.gcf(), 'completeness_heatmap.png')

print("\nEnhanced cluster analysis completed. Results saved to 'cluster_analysis_results' folder.")
