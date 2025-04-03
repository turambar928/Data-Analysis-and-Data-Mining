import time
from sklearn.mixture import GaussianMixture
from ucimlrepo import fetch_ucirepo
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Fetch and prepare data
iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dictionary to store runtimes
runtimes = {}

# Test each algorithm
n_clusters = 3

# 1. Hierarchical Clustering
start = time.time()
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
hierarchical.fit_predict(X_scaled)
runtimes['Hierarchical'] = time.time() - start

# 2. Spectral Clustering
start = time.time()
spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=1.0, random_state=42)
spectral.fit_predict(X_scaled)
runtimes['Spectral'] = time.time() - start

# 3. K-Means
start = time.time()
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
runtimes['K-Means'] = time.time() - start

# 4. Expectation Maximization (GMM)
start = time.time()
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(X_scaled)
runtimes['GMM (EM)'] = time.time() - start

# 5. DBSCAN
start = time.time()
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit_predict(X_scaled)
runtimes['DBSCAN'] = time.time() - start

# Display results
print("Algorithm Runtimes (seconds):")
for algo, runtime in runtimes.items():
    print(f"{algo}: {runtime:.4f}")

# Plot comparison
plt.figure(figsize=(10, 6))
plt.bar(runtimes.keys(), runtimes.values(), color=['blue', 'green', 'red', 'purple', 'orange'])
plt.title('Comparative Runtime of Clustering Algorithms on Iris Dataset')
plt.ylabel('Execution Time (seconds)')
plt.yscale('log')  # Use log scale if time differences are large
plt.show()