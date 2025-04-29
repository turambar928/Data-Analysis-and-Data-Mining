import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

# Set plot style
sns.set_style('whitegrid')  # seaborn 样式设置
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Create output directory for visualizations
os.makedirs('association_rules_results', exist_ok=True)

# Function to save figures
def save_fig(fig, filename):
    fig.savefig(os.path.join('association_rules_results', filename), bbox_inches='tight', dpi=300)
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
    'HIS出院科室': 'Department',
    '骨钙素(N-MID)': 'Osteocalcin',
    '总I型胶原氨基端延长肽': 'Total Type I Collagen N-terminal Propeptide',
    '促甲状腺素': 'TSH',
    '促甲状腺素受体抗体': 'TSH Receptor Antibody',
    '甲状腺过氧化物酶抗体': 'Thyroid Peroxidase Antibody',
    '甲状腺球蛋白': 'Thyroglobulin',
    '甲状腺球蛋白抗体': 'Thyroglobulin Antibody',
    '总四碘甲状腺原氨酸': 'Total T4',
    '高密度脂蛋白': 'HDL',
    '25-羟基维生素D': 'Vitamin D (25-OH)',
    '碱性磷酸酶': 'Alkaline Phosphatase',
    '骨源碱性磷酸酶': 'Bone-specific Alkaline Phosphatase',
    '降钙素': 'Calcitonin',
    '尿钙': 'Urinary Calcium',
    '天门冬氨酸转氨酶': 'AST',
    'γ-谷氨酰转肽酶': 'GGT',
    '总胆红素': 'Total Bilirubin',
    '直接胆红素': 'Direct Bilirubin',
    '胰岛素': 'Insulin',
    'C肽': 'C-peptide',
    'β-胶原特殊序列': 'β-Collagen Special Sequence',
    '雌二醇': 'Estradiol',
    '促卵泡成熟素': 'FSH',
    '促黄体生成素': 'LH',
    '泌乳素': 'Prolactin',
    '孕酮': 'Progesterone',
    '睾酮': 'Testosterone',
    '硫酸去氢表雄酮': 'DHEA-S',
    '叶酸': 'Folic Acid',
    '维生素B12': 'Vitamin B12'
}

# Department translations
dept_mapping = {
    '内分泌代谢科': 'Endocrinology & Metabolism',
    '惠宾病房': 'VIP Ward',
    '日间病房': 'Day Ward'
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

# 1. Prepare data for association rule mining
print("\n=== PREPARING DATA FOR ASSOCIATION RULE MINING ===")

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
    '葡萄糖': (3.9, 6.1),  # mmol/L
    '游离甲状腺素': (12.0, 22.0),  # pmol/L
    '总三碘甲状腺原氨酸': (1.3, 3.1),  # nmol/L
    '低密度脂蛋白': (0, 3.12),  # mmol/L
    '总胆固醇': (0, 5.2),  # mmol/L
    '甘油三酯': (0, 1.7),  # mmol/L
    '高密度脂蛋白': (1.29, 1.55),  # mmol/L
    '促甲状腺素': (0.27, 4.2),  # mIU/L
    '骨钙素(N-MID)': (11.0, 46.0),  # ng/mL
    '甲状旁腺激素': (1.6, 6.9)  # pmol/L
}

# Create binary features for abnormal lab values
binary_features = []

for lab, (lower, upper) in normal_ranges.items():
    if lab in df.columns:
        # Convert to numeric
        df[lab] = pd.to_numeric(df[lab], errors='coerce')
        
        # Create binary features
        low_col = f'{lab}_Low'
        high_col = f'{lab}_High'
        
        df[low_col] = (df[lab] < lower).astype(int)
        df[high_col] = (df[lab] > upper).astype(int)
        
        binary_features.extend([low_col, high_col])

print(f"Created {len(binary_features)} binary features for abnormal lab values")

# Add department as a feature
if 'HIS出院科室' in df.columns:
    for dept in df['HIS出院科室'].unique():
        col_name = f'Dept_{dept}'
        df[col_name] = (df['HIS出院科室'] == dept).astype(int)
        binary_features.append(col_name)

# Add length of stay categories
if '住院天数' in df.columns:
    df['LOS_Short'] = (df['住院天数'] < 5).astype(int)
    df['LOS_Medium'] = ((df['住院天数'] >= 5) & (df['住院天数'] < 10)).astype(int)
    df['LOS_Long'] = (df['住院天数'] >= 10).astype(int)
    binary_features.extend(['LOS_Short', 'LOS_Medium', 'LOS_Long'])

# 2. Association Rule Mining
print("\n=== ASSOCIATION RULE MINING ===")

# Group by patient to get one record per patient
if '住院号码' in df.columns:
    # For each patient, take the maximum value of each binary feature
    # (1 means the condition was present at least once)
    patient_df = df.groupby('住院号码')[binary_features].max()
    
    print(f"Prepared data for {len(patient_df)} patients with {len(binary_features)} features")
    
    # Apply Apriori algorithm
    print("Applying Apriori algorithm...")
    frequent_itemsets = apriori(patient_df, min_support=0.05, use_colnames=True)
    
    if not frequent_itemsets.empty:
        print(f"Found {len(frequent_itemsets)} frequent itemsets")
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        
        if not rules.empty:
            print(f"Generated {len(rules)} association rules")
            
            # Sort rules by lift
            rules = rules.sort_values('lift', ascending=False)
            
            # Save rules to CSV
            rules.to_csv('association_rules_results/association_rules.csv', index=False)
            
            # Print top rules
            print("\nTop 10 Association Rules by Lift:")
            for i, row in rules.head(10).iterrows():
                antecedents = ', '.join(list(row['antecedents']))
                consequents = ', '.join(list(row['consequents']))
                print(f"{antecedents} => {consequents} (Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f})")
            
            # Visualize top rules
            plt.figure(figsize=(10, 8))
            plt.scatter(rules['support'], rules['confidence'], alpha=0.5, s=rules['lift']*20)
            plt.xlabel('Support', fontsize=12)
            plt.ylabel('Confidence', fontsize=12)
            plt.title('Association Rules - Support vs Confidence', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            save_fig(plt.gcf(), 'rules_support_confidence.png')
            
            # Create network visualization of top rules
            G = nx.DiGraph()
            
            # Add nodes and edges for top 20 rules
            for i, row in rules.head(20).iterrows():
                antecedents = frozenset(row['antecedents'])
                consequents = frozenset(row['consequents'])
                
                # Add nodes
                for item in antecedents:
                    if item not in G:
                        # Clean up the feature name for display
                        display_name = item
                        if '_Low' in item:
                            lab = item.replace('_Low', '')
                            display_name = f"Low {lab_tests_en.get(lab, lab)}"
                        elif '_High' in item:
                            lab = item.replace('_High', '')
                            display_name = f"High {lab_tests_en.get(lab, lab)}"
                        elif 'Dept_' in item:
                            dept = item.replace('Dept_', '')
                            display_name = f"Dept: {dept_mapping.get(dept, dept)}"
                        
                        G.add_node(item, label=display_name)
                
                for item in consequents:
                    if item not in G:
                        # Clean up the feature name for display
                        display_name = item
                        if '_Low' in item:
                            lab = item.replace('_Low', '')
                            display_name = f"Low {lab_tests_en.get(lab, lab)}"
                        elif '_High' in item:
                            lab = item.replace('_High', '')
                            display_name = f"High {lab_tests_en.get(lab, lab)}"
                        elif 'Dept_' in item:
                            dept = item.replace('Dept_', '')
                            display_name = f"Dept: {dept_mapping.get(dept, dept)}"
                        
                        G.add_node(item, label=display_name)
                
                # Add edge for the rule
                for a in antecedents:
                    for c in consequents:
                        G.add_edge(a, c, weight=row['lift'], confidence=row['confidence'])
            
            # Plot the network
            plt.figure(figsize=(14, 12))
            
            # Use spring layout
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8)
            
            # Draw edges with width based on lift
            edges = G.edges()
            weights = [G[u][v]['weight'] * 0.5 for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, edge_color='darkblue', 
                                  arrowsize=20, connectionstyle='arc3,rad=0.1')
            
            # Draw labels
            labels = nx.get_node_attributes(G, 'label')
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family='sans-serif')
            
            plt.axis('off')
            plt.title('Association Rule Network (Top 20 Rules)', fontsize=16)
            save_fig(plt.gcf(), 'rules_network.png')
            
            # 3. Focused Analysis: Thyroid-Bone Association
            print("\n=== FOCUSED ANALYSIS: THYROID-BONE ASSOCIATION ===")
            
            # Define thyroid and bone-related features
            thyroid_features = [f for f in binary_features if any(x in f for x in 
                              ['甲状腺', '促甲状腺', 'T3', 'T4'])]
            
            bone_features = [f for f in binary_features if any(x in f for x in 
                           ['钙', '磷', '骨钙素', '碱性磷酸酶', '骨源', '甲状旁腺'])]
            
            print(f"Found {len(thyroid_features)} thyroid-related features and {len(bone_features)} bone-related features")
            
            # Filter rules related to thyroid and bone
            thyroid_bone_rules = rules[
                rules['antecedents'].apply(lambda x: any(item in thyroid_features for item in x)) & 
                rules['consequents'].apply(lambda x: any(item in bone_features for item in x))
            ]
            
            bone_thyroid_rules = rules[
                rules['antecedents'].apply(lambda x: any(item in bone_features for item in x)) & 
                rules['consequents'].apply(lambda x: any(item in thyroid_features for item in x))
            ]
            
            thyroid_bone_all = pd.concat([thyroid_bone_rules, bone_thyroid_rules])
            
            if not thyroid_bone_all.empty:
                print(f"Found {len(thyroid_bone_all)} rules connecting thyroid and bone markers")
                
                # Save these specific rules
                thyroid_bone_all.to_csv('association_rules_results/thyroid_bone_rules.csv', index=False)
                
                # Print top thyroid-bone rules
                print("\nTop Thyroid-Bone Association Rules:")
                for i, row in thyroid_bone_all.head(5).iterrows():
                    antecedents = ', '.join(list(row['antecedents']))
                    consequents = ', '.join(list(row['consequents']))
                    print(f"{antecedents} => {consequents} (Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f})")
                
                # Create a focused network visualization
                G_tb = nx.DiGraph()
                
                # Add nodes and edges
                for i, row in thyroid_bone_all.iterrows():
                    antecedents = frozenset(row['antecedents'])
                    consequents = frozenset(row['consequents'])
                    
                    # Add nodes
                    for item in antecedents:
                        if item not in G_tb:
                            # Clean up the feature name for display
                            display_name = item
                            if '_Low' in item:
                                lab = item.replace('_Low', '')
                                display_name = f"Low {lab_tests_en.get(lab, lab)}"
                            elif '_High' in item:
                                lab = item.replace('_High', '')
                                display_name = f"High {lab_tests_en.get(lab, lab)}"
                            
                            # Determine node type (thyroid or bone)
                            node_type = 'thyroid' if item in thyroid_features else 'bone'
                            G_tb.add_node(item, label=display_name, type=node_type)
                    
                    for item in consequents:
                        if item not in G_tb:
                            # Clean up the feature name for display
                            display_name = item
                            if '_Low' in item:
                                lab = item.replace('_Low', '')
                                display_name = f"Low {lab_tests_en.get(lab, lab)}"
                            elif '_High' in item:
                                lab = item.replace('_High', '')
                                display_name = f"High {lab_tests_en.get(lab, lab)}"
                            
                            # Determine node type (thyroid or bone)
                            node_type = 'thyroid' if item in thyroid_features else 'bone'
                            G_tb.add_node(item, label=display_name, type=node_type)
                    
                    # Add edge for the rule
                    for a in antecedents:
                        for c in consequents:
                            G_tb.add_edge(a, c, weight=row['lift'], confidence=row['confidence'])
                
                # Plot the network
                plt.figure(figsize=(14, 12))
                
                # Use spring layout
                pos = nx.spring_layout(G_tb, k=0.5, iterations=50)
                
                # Draw nodes with different colors for thyroid and bone
                thyroid_nodes = [n for n, attr in G_tb.nodes(data=True) if attr.get('type') == 'thyroid']
                bone_nodes = [n for n, attr in G_tb.nodes(data=True) if attr.get('type') == 'bone']
                
                nx.draw_networkx_nodes(G_tb, pos, nodelist=thyroid_nodes, node_size=2000, 
                                      node_color='lightblue', alpha=0.8, label='Thyroid')
                nx.draw_networkx_nodes(G_tb, pos, nodelist=bone_nodes, node_size=2000, 
                                      node_color='lightgreen', alpha=0.8, label='Bone')
                
                # Draw edges with width based on lift
                edges = G_tb.edges()
                weights = [G_tb[u][v]['weight'] * 0.5 for u, v in edges]
                nx.draw_networkx_edges(G_tb, pos, width=weights, alpha=0.7, edge_color='darkblue', 
                                      arrowsize=20, connectionstyle='arc3,rad=0.1')
                
                # Draw labels
                labels = nx.get_node_attributes(G_tb, 'label')
                nx.draw_networkx_labels(G_tb, pos, labels=labels, font_size=10, font_family='sans-serif')
                
                plt.axis('off')
                plt.title('Thyroid-Bone Association Network', fontsize=16)
                plt.legend()
                save_fig(plt.gcf(), 'thyroid_bone_network.png')
            
            # 4. Comorbidity Analysis
            print("\n=== COMORBIDITY ANALYSIS ===")
            
            # Define high values that indicate potential disorders
            disorders = {
                '葡萄糖_High': 'Hyperglycemia',
                '糖化血红蛋白_High': 'Diabetes',
                '肌酐_High': 'Renal Dysfunction',
                '尿素_High': 'Azotemia',
                '尿酸_High': 'Hyperuricemia',
                '总胆固醇_High': 'Hypercholesterolemia',
                '低密度脂蛋白_High': 'High LDL',
                '甘油三酯_High': 'Hypertriglyceridemia',
                '游离甲状腺素_High': 'Hyperthyroidism',
                '游离甲状腺素_Low': 'Hypothyroidism',
                '总三碘甲状腺原氨酸_High': 'T3 Elevation',
                '总三碘甲状腺原氨酸_Low': 'T3 Deficiency',
                '钙_High': 'Hypercalcemia',
                '钙_Low': 'Hypocalcemia',
                '磷_High': 'Hyperphosphatemia',
                '磷_Low': 'Hypophosphatemia',
                '钾_High': 'Hyperkalemia',
                '钾_Low': 'Hypokalemia',
                '钠_High': 'Hypernatremia',
                '钠_Low': 'Hyponatremia'
            }
            
            # Count co-occurrences of disorders
            disorder_cols = [col for col in binary_features if col in disorders]
            
            if disorder_cols:
                # Create a co-occurrence matrix
                comorbidity_df = patient_df[disorder_cols]
                comorbidity_matrix = np.zeros((len(disorder_cols), len(disorder_cols)))
                
                for i, disorder1 in enumerate(disorder_cols):
                    for j, disorder2 in enumerate(disorder_cols):
                        if i != j:
                            # Count co-occurrences
                            comorbidity_matrix[i, j] = ((comorbidity_df[disorder1] == 1) & 
                                                       (comorbidity_df[disorder2] == 1)).sum()
                
                # Create a DataFrame for the matrix
                comorbidity_df = pd.DataFrame(comorbidity_matrix, 
                                             index=[disorders[d] for d in disorder_cols],
                                             columns=[disorders[d] for d in disorder_cols])
                
                # Save to CSV
                comorbidity_df.to_csv('association_rules_results/comorbidity_matrix.csv')
                
                # Plot heatmap
                plt.figure(figsize=(14, 12))
                sns.heatmap(comorbidity_df, annot=True, fmt='g', cmap='YlGnBu')
                plt.title('Comorbidity Matrix (Co-occurrence Counts)', fontsize=16)
                plt.tight_layout()
                save_fig(plt.gcf(), 'comorbidity_heatmap.png')
                
                # Calculate comorbidity network
                # Normalize by the minimum prevalence of each disorder
                normalized_matrix = np.zeros_like(comorbidity_matrix)
                
                for i, disorder1 in enumerate(disorder_cols):
                    for j, disorder2 in enumerate(disorder_cols):
                        if i != j:
                            # Get prevalence of each disorder
                            prev1 = comorbidity_df[disorder1].sum()
                            prev2 = comorbidity_df[disorder2].sum()
                            
                            if min(prev1, prev2) > 0:
                                # Normalize by minimum prevalence
                                normalized_matrix[i, j] = comorbidity_matrix[i, j] / min(prev1, prev2)
                
                # Create a network from the normalized matrix
                G_comorbid = nx.DiGraph()
                
                # Add nodes
                for i, disorder in enumerate(disorder_cols):
                    G_comorbid.add_node(disorders[disorder])
                
                # Add edges for strong associations (threshold can be adjusted)
                threshold = 0.1  # Minimum normalized co-occurrence
                
                for i, disorder1 in enumerate(disorder_cols):
                    for j, disorder2 in enumerate(disorder_cols):
                        if i != j and normalized_matrix[i, j] > threshold:
                            G_comorbid.add_edge(disorders[disorder1], disorders[disorder2], 
                                              weight=normalized_matrix[i, j])
                
                # Plot the network
                plt.figure(figsize=(16, 14))
                
                # Use spring layout
                pos = nx.spring_layout(G_comorbid, k=0.3, iterations=50)
                
                # Draw nodes
                nx.draw_networkx_nodes(G_comorbid, pos, node_size=2500, node_color='lightblue', alpha=0.8)
                
                # Draw edges with width based on weight
                edges = G_comorbid.edges()
                if edges:
                    weights = [G_comorbid[u][v]['weight'] * 3 for u, v in edges]
                    nx.draw_networkx_edges(G_comorbid, pos, width=weights, alpha=0.7, 
                                          edge_color='darkblue', arrowstyle='-')
                
                # Draw labels
                nx.draw_networkx_labels(G_comorbid, pos, font_size=10, font_family='sans-serif')
                
                plt.axis('off')
                plt.title('Comorbidity Network', fontsize=16)
                save_fig(plt.gcf(), 'comorbidity_network.png')

print("\nClinical association rule mining completed. Results saved to 'association_rules_results' folder.")
