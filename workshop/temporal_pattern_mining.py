import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import os
from datetime import datetime, timedelta
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('whitegrid')  # seaborn 样式设置
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Create output directory for visualizations
os.makedirs('temporal_analysis_results', exist_ok=True)

# Function to save figures
def save_fig(fig, filename):
    fig.savefig(os.path.join('temporal_analysis_results', filename), bbox_inches='tight', dpi=300)
    plt.close(fig)

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

print("Loading dataset...")
file_path = "anonymized data for workshop.xlsx"
df = pd.read_excel(file_path)
print(f"Dataset loaded with shape: {df.shape}")

# Convert date columns to datetime if not already
date_columns = ['入院时间', '出院时间', '采集时间']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# 1. Time Series Clustering
print("\n=== TIME SERIES CLUSTERING ===")

# Check if we have patient IDs and timestamps
if '住院号码' in df.columns and '采集时间' in df.columns:
    # Get patients with multiple measurements
    patient_counts = df['住院号码'].value_counts()
    patients_with_multiple = patient_counts[patient_counts > 5].index.tolist()
    
    if patients_with_multiple:
        print(f"Number of patients with more than 5 measurements: {len(patients_with_multiple)}")
        
        # Select key lab tests for time series analysis
        ts_labs = ['钾', '钠', '氯', '钙', '磷', '肌酐', '尿素', '葡萄糖', '糖化血红蛋白']
        ts_labs = [lab for lab in ts_labs if lab in df.columns]
        
        if ts_labs:
            # Focus on glucose for time series clustering as an example
            focus_lab = '葡萄糖'  # Glucose
            if focus_lab in ts_labs:
                print(f"\nPerforming time series clustering for {lab_tests_en[focus_lab]}")
                
                # Prepare time series data
                patient_series = {}
                for patient in patients_with_multiple[:100]:  # Limit to first 100 patients for computational efficiency
                    patient_data = df[df['住院号码'] == patient].sort_values('采集时间')
                    # Convert to numeric and handle missing values
                    patient_data[focus_lab] = pd.to_numeric(patient_data[focus_lab], errors='coerce')
                    # Only include if we have at least 5 measurements
                    if patient_data[focus_lab].notna().sum() >= 5:
                        # Get the time series
                        values = patient_data[focus_lab].dropna().values
                        if len(values) >= 5:
                            patient_series[patient] = values
                
                if patient_series:
                    print(f"Found {len(patient_series)} patients with sufficient {lab_tests_en[focus_lab]} measurements")
                    
                    # Pad series to the same length
                    max_length = max(len(series) for series in patient_series.values())
                    padded_series = []
                    patient_ids = []
                    
                    for patient, series in patient_series.items():
                        # Pad with NaN
                        padded = np.full(max_length, np.nan)
                        padded[:len(series)] = series
                        padded_series.append(padded)
                        patient_ids.append(patient)
                    
                    # Convert to numpy array
                    ts_data = np.array(padded_series)
                    
                    # Handle NaN values (replace with mean of the series)
                    for i, series in enumerate(ts_data):
                        mask = np.isnan(series)
                        series[mask] = np.mean(series[~mask])
                    
                    # Scale the time series
                    scaler = TimeSeriesScalerMeanVariance()
                    ts_data_scaled = scaler.fit_transform(ts_data)
                    
                    # Determine optimal number of clusters
                    max_clusters = min(10, len(ts_data) // 5)  # Limit based on data size
                    if max_clusters >= 2:
                        inertia = []
                        silhouette_scores = []
                        
                        for k in range(2, max_clusters + 1):
                            # Use DTW (Dynamic Time Warping) for time series clustering
                            km = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42)
                            labels = km.fit_predict(ts_data_scaled)
                            inertia.append(km.inertia_)
                            
                            # Calculate silhouette score if we have more than one cluster
                            if len(np.unique(labels)) > 1:
                                score = silhouette_score(ts_data_scaled.reshape(ts_data_scaled.shape[0], -1), labels)
                                silhouette_scores.append(score)
                            else:
                                silhouette_scores.append(0)
                        
                        # Plot elbow curve
                        plt.figure(figsize=(10, 6))
                        plt.plot(range(2, max_clusters + 1), inertia, 'o-')
                        plt.title(f'Elbow Method for Optimal k in {lab_tests_en[focus_lab]} Time Series', fontsize=14)
                        plt.xlabel('Number of Clusters (k)', fontsize=12)
                        plt.ylabel('Inertia', fontsize=12)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        save_fig(plt.gcf(), f'ts_elbow_curve_{focus_lab}.png')
                        
                        # Plot silhouette scores
                        plt.figure(figsize=(10, 6))
                        plt.plot(range(2, max_clusters + 1), silhouette_scores, 'o-')
                        plt.title(f'Silhouette Scores for {lab_tests_en[focus_lab]} Time Series Clustering', fontsize=14)
                        plt.xlabel('Number of Clusters (k)', fontsize=12)
                        plt.ylabel('Silhouette Score', fontsize=12)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        save_fig(plt.gcf(), f'ts_silhouette_{focus_lab}.png')
                        
                        # Choose optimal k (either from elbow or silhouette)
                        optimal_k = np.argmax(silhouette_scores) + 2  # +2 because we started from k=2
                        print(f"Optimal number of time series clusters: {optimal_k}")
                        
                        # Perform clustering with optimal k
                        km = TimeSeriesKMeans(n_clusters=optimal_k, metric="dtw", random_state=42)
                        labels = km.fit_predict(ts_data_scaled)
                        
                        # Create a DataFrame with cluster assignments
                        ts_clusters = pd.DataFrame({
                            'Patient_ID': patient_ids,
                            'TS_Cluster': labels
                        })
                        
                        # Save cluster assignments
                        ts_clusters.to_csv('temporal_analysis_results/time_series_clusters.csv', index=False)
                        
                        # Plot representative time series for each cluster
                        plt.figure(figsize=(15, 10))
                        for i in range(optimal_k):
                            cluster_series = ts_data[labels == i]
                            
                            # Plot individual series with low alpha
                            for series in cluster_series:
                                plt.plot(series, 'k-', alpha=0.1)
                            
                            # Plot the cluster center with high alpha
                            center = np.mean(cluster_series, axis=0)
                            plt.plot(center, linewidth=2, label=f'Cluster {i+1} (n={len(cluster_series)})')
                        
                        plt.title(f'{lab_tests_en[focus_lab]} Time Series Patterns', fontsize=16)
                        plt.xlabel('Time Point', fontsize=12)
                        plt.ylabel(f'{lab_tests_en[focus_lab]} Value', fontsize=12)
                        plt.legend()
                        plt.grid(True, linestyle='--', alpha=0.7)
                        save_fig(plt.gcf(), f'ts_patterns_{focus_lab}.png')
                        
                        # Plot individual clusters
                        for i in range(optimal_k):
                            plt.figure(figsize=(12, 6))
                            cluster_series = ts_data[labels == i]
                            
                            # Plot individual series with low alpha
                            for series in cluster_series:
                                plt.plot(series, 'k-', alpha=0.2)
                            
                            # Plot the cluster center with high alpha
                            center = np.mean(cluster_series, axis=0)
                            plt.plot(center, 'r-', linewidth=3)
                            
                            plt.title(f'{lab_tests_en[focus_lab]} Pattern - Cluster {i+1} (n={len(cluster_series)})', fontsize=14)
                            plt.xlabel('Time Point', fontsize=12)
                            plt.ylabel(f'{lab_tests_en[focus_lab]} Value', fontsize=12)
                            plt.grid(True, linestyle='--', alpha=0.7)
                            save_fig(plt.gcf(), f'ts_cluster_{focus_lab}_{i+1}.png')

# 2. State Transition Analysis
print("\n=== STATE TRANSITION ANALYSIS ===")

# Define states based on lab values
def define_states(row, lab, thresholds):
    if pd.isna(row[lab]):
        return np.nan
    
    value = pd.to_numeric(row[lab], errors='coerce')
    if pd.isna(value):
        return np.nan
    
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return i
    
    return len(thresholds)

# Define thresholds for key labs
state_thresholds = {
    '葡萄糖': [5.6, 7.0, 11.1],  # Normal, Prediabetes, Diabetes, Severe
    '糖化血红蛋白': [5.7, 6.5, 8.0],  # Normal, Prediabetes, Diabetes, Poorly controlled
    '肌酐': [84, 115, 150]  # Normal, Mild, Moderate, Severe
}

# State names for plotting
state_names = {
    '葡萄糖': ['Normal', 'Prediabetes', 'Diabetes', 'Severe Hyperglycemia'],
    '糖化血红蛋白': ['Normal', 'Prediabetes', 'Diabetes', 'Poorly Controlled'],
    '肌酐': ['Normal', 'Mild Elevation', 'Moderate Elevation', 'Severe Elevation']
}

# Check if we have patient IDs and timestamps
if '住院号码' in df.columns and '采集时间' in df.columns:
    # Get patients with multiple measurements
    patient_counts = df['住院号码'].value_counts()
    patients_with_multiple = patient_counts[patient_counts > 3].index.tolist()
    
    if patients_with_multiple:
        print(f"Number of patients with more than 3 measurements: {len(patients_with_multiple)}")
        
        # Select labs for state transition analysis
        for lab, thresholds in state_thresholds.items():
            if lab in df.columns:
                print(f"\nPerforming state transition analysis for {lab_tests_en[lab]}")
                
                # Define states for this lab
                df[f'{lab}_State'] = df.apply(lambda row: define_states(row, lab, thresholds), axis=1)
                
                # Count transitions between states
                transitions = {}
                patient_trajectories = {}
                
                for patient in patients_with_multiple:
                    patient_data = df[df['住院号码'] == patient].sort_values('采集时间')
                    states = patient_data[f'{lab}_State'].dropna().astype(int).tolist()
                    
                    if len(states) > 1:
                        # Record the trajectory
                        patient_trajectories[patient] = states
                        
                        # Count transitions
                        for i in range(len(states) - 1):
                            from_state = states[i]
                            to_state = states[i + 1]
                            
                            if (from_state, to_state) not in transitions:
                                transitions[(from_state, to_state)] = 0
                            
                            transitions[(from_state, to_state)] += 1
                
                if transitions:
                    # Create transition matrix
                    num_states = len(thresholds) + 1
                    transition_matrix = np.zeros((num_states, num_states))
                    
                    for (from_state, to_state), count in transitions.items():
                        transition_matrix[from_state, to_state] = count
                    
                    # Normalize by row sums to get probabilities
                    row_sums = transition_matrix.sum(axis=1)
                    row_sums[row_sums == 0] = 1  # Avoid division by zero
                    transition_probs = transition_matrix / row_sums[:, np.newaxis]
                    
                    # Plot transition matrix
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(transition_probs, annot=True, cmap='YlGnBu', fmt='.2f',
                                xticklabels=state_names[lab],
                                yticklabels=state_names[lab])
                    plt.title(f'State Transition Probabilities for {lab_tests_en[lab]}', fontsize=14)
                    plt.xlabel('To State', fontsize=12)
                    plt.ylabel('From State', fontsize=12)
                    plt.tight_layout()
                    save_fig(plt.gcf(), f'transition_matrix_{lab}.png')
                    
                    # Save transition matrix
                    transition_df = pd.DataFrame(transition_probs, 
                                                index=state_names[lab],
                                                columns=state_names[lab])
                    transition_df.to_csv(f'temporal_analysis_results/transition_matrix_{lab}.csv')
                    
                    # Analyze common trajectories
                    trajectory_counts = {}
                    for trajectory in patient_trajectories.values():
                        trajectory_tuple = tuple(trajectory)
                        if trajectory_tuple not in trajectory_counts:
                            trajectory_counts[trajectory_tuple] = 0
                        trajectory_counts[trajectory_tuple] += 1
                    
                    # Sort by frequency
                    sorted_trajectories = sorted(trajectory_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    # Print and save top trajectories
                    print(f"\nTop 5 {lab_tests_en[lab]} Trajectories:")
                    trajectory_data = []
                    
                    for i, (trajectory, count) in enumerate(sorted_trajectories[:5]):
                        trajectory_str = ' → '.join([state_names[lab][state] for state in trajectory])
                        print(f"{i+1}. {trajectory_str}: {count} patients")
                        
                        trajectory_data.append({
                            'Trajectory': trajectory_str,
                            'Count': count,
                            'Percentage': count / len(patient_trajectories) * 100
                        })
                    
                    # Save trajectory data
                    trajectory_df = pd.DataFrame(trajectory_data)
                    trajectory_df.to_csv(f'temporal_analysis_results/trajectories_{lab}.csv', index=False)
                    
                    # Plot top trajectories
                    plt.figure(figsize=(12, 6))
                    top_n = min(10, len(sorted_trajectories))
                    trajectories = [' → '.join([state_names[lab][state] for state in traj[0]]) for traj in sorted_trajectories[:top_n]]
                    counts = [traj[1] for traj in sorted_trajectories[:top_n]]
                    
                    plt.barh(range(len(trajectories)), counts, align='center')
                    plt.yticks(range(len(trajectories)), trajectories)
                    plt.xlabel('Number of Patients', fontsize=12)
                    plt.title(f'Most Common {lab_tests_en[lab]} Trajectories', fontsize=14)
                    plt.tight_layout()
                    save_fig(plt.gcf(), f'common_trajectories_{lab}.png')

# 3. Predictive Modeling for Lab Values
print("\n=== PREDICTIVE MODELING FOR LAB VALUES ===")

# This would typically involve time series forecasting models
# For simplicity, we'll just analyze the rate of change in lab values

# Check if we have patient IDs and timestamps
if '住院号码' in df.columns and '采集时间' in df.columns:
    # Get patients with multiple measurements
    patient_counts = df['住院号码'].value_counts()
    patients_with_multiple = patient_counts[patient_counts > 2].index.tolist()
    
    if patients_with_multiple:
        print(f"Number of patients with more than 2 measurements: {len(patients_with_multiple)}")
        
        # Select key lab tests for analysis
        pred_labs = ['葡萄糖', '糖化血红蛋白', '肌酐']
        pred_labs = [lab for lab in pred_labs if lab in df.columns]
        
        for lab in pred_labs:
            print(f"\nAnalyzing rate of change for {lab_tests_en[lab]}")
            
            # Calculate rate of change for each patient
            rate_of_change = []
            
            for patient in patients_with_multiple:
                patient_data = df[df['住院号码'] == patient].sort_values('采集时间')
                
                # Convert to numeric
                patient_data[lab] = pd.to_numeric(patient_data[lab], errors='coerce')
                
                # Drop missing values
                patient_data = patient_data.dropna(subset=[lab])
                
                if len(patient_data) >= 2:
                    # Calculate time differences in days
                    time_diffs = [(t - patient_data['采集时间'].iloc[0]).total_seconds() / (24 * 3600) 
                                 for t in patient_data['采集时间']]
                    
                    # Calculate value differences
                    value_diffs = [v - patient_data[lab].iloc[0] for v in patient_data[lab]]
                    
                    # Calculate rate of change (per day)
                    if time_diffs[-1] > 0:
                        overall_rate = value_diffs[-1] / time_diffs[-1]
                        
                        rate_of_change.append({
                            'Patient_ID': patient,
                            'Initial_Value': patient_data[lab].iloc[0],
                            'Final_Value': patient_data[lab].iloc[-1],
                            'Days_Between': time_diffs[-1],
                            'Rate_Per_Day': overall_rate
                        })
            
            if rate_of_change:
                # Create DataFrame
                rate_df = pd.DataFrame(rate_of_change)
                
                # Save to CSV
                rate_df.to_csv(f'temporal_analysis_results/rate_of_change_{lab}.csv', index=False)
                
                # Plot distribution of rates
                plt.figure(figsize=(10, 6))
                sns.histplot(rate_df['Rate_Per_Day'], kde=True)
                plt.title(f'Distribution of {lab_tests_en[lab]} Rate of Change (per day)', fontsize=14)
                plt.xlabel('Rate of Change (per day)', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                save_fig(plt.gcf(), f'rate_distribution_{lab}.png')
                
                # Plot rate vs initial value
                plt.figure(figsize=(10, 6))
                plt.scatter(rate_df['Initial_Value'], rate_df['Rate_Per_Day'], alpha=0.5)
                plt.title(f'Rate of Change vs Initial {lab_tests_en[lab]} Value', fontsize=14)
                plt.xlabel(f'Initial {lab_tests_en[lab]} Value', fontsize=12)
                plt.ylabel('Rate of Change (per day)', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add trend line
                if len(rate_df) > 1:
                    z = np.polyfit(rate_df['Initial_Value'], rate_df['Rate_Per_Day'], 1)
                    p = np.poly1d(z)
                    plt.plot(rate_df['Initial_Value'], p(rate_df['Initial_Value']), "r--")
                
                save_fig(plt.gcf(), f'rate_vs_initial_{lab}.png')
                
                # Calculate summary statistics
                print(f"\n{lab_tests_en[lab]} Rate of Change Statistics:")
                print(f"Mean rate: {rate_df['Rate_Per_Day'].mean():.4f} per day")
                print(f"Median rate: {rate_df['Rate_Per_Day'].median():.4f} per day")
                print(f"Std deviation: {rate_df['Rate_Per_Day'].std():.4f}")
                
                # Identify patients with significant improvement or deterioration
                threshold = rate_df['Rate_Per_Day'].std()
                
                improving = rate_df[rate_df['Rate_Per_Day'] < -threshold]
                stable = rate_df[(rate_df['Rate_Per_Day'] >= -threshold) & (rate_df['Rate_Per_Day'] <= threshold)]
                worsening = rate_df[rate_df['Rate_Per_Day'] > threshold]
                
                print(f"Patients with significant improvement: {len(improving)} ({len(improving)/len(rate_df)*100:.1f}%)")
                print(f"Patients with stable values: {len(stable)} ({len(stable)/len(rate_df)*100:.1f}%)")
                print(f"Patients with significant deterioration: {len(worsening)} ({len(worsening)/len(rate_df)*100:.1f}%)")

print("\nTemporal pattern mining completed. Results saved to 'temporal_analysis_results' folder.")
