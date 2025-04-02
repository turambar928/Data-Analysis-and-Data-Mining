import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import time


df = pd.read_excel("Raisin_3.xlsx")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

'''
linkage methods
'''

linkage_methods = ['single', 'complete', 'average', 'ward']
result = {}

for method in linkage_methods:
    print(f"Process {method} linkage...")
    start_time = time.time()
    Z = linkage(X_scaled, method=method)
    computation_time = time.time() - start_time
    result[method] = {
        'linkage_matrix': Z,
        'time': computation_time
    }

plt.figure(figsize=(15, 10))
for i,method in enumerate(linkage_methods, 1):
    plt.subplot(2, 2, i)
    plt.title(f'Dendrogram {method} linkage')
    dendrogram(result[method]['linkage_matrix'],
               truncate_mode='lastp',
               p=20,
               show_leaf_counts=True,
               leaf_rotation=90.,
               leaf_font_size=12.,
               show_contracted=True)
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')

plt.tight_layout()
plt.show()


#评估不同方法
performance_df = pd.DataFrame(columns=['Method', 'Time (s)', 'Silhouette Score'])
for method in linkage_methods:
    Z = result[method]['linkage_matrix']
    clusters = fcluster(Z, t=2, criterion='maxclust')
    if len(X_scaled) > 5000:
        sample_idx = np.random.choice(len(X_scaled), 5000, replace=False)
        score = silhouette_score(X_scaled[sample_idx], clusters[sample_idx])
    else:
        score = silhouette_score(X_scaled, clusters)

    performance_df = performance_df.append({
        'Method': method,
        'Time (s)': result[method]['time'],
        'Silhouette Score': score
    }, ignore_index=True)

    print("\nPerformance Comparison:")
    print(performance_df)







