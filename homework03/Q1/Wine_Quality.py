import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time

# Load and prepare data
data = pd.read_csv('winequality-white.csv', sep=';')
X = data.drop('quality', axis=1)
y = data['quality']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Function to plot clustering results
def plot_clusters(X, labels, title, centers=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50, alpha=0.6)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Cluster')
    plt.grid(True)
    if centers is not None:
        plt.legend()
    plt.show()

# Determine optimal eps for DBSCAN
def find_optimal_eps(X, neighbors=5):
    neigh = NearestNeighbors(n_neighbors=neighbors)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances[:, -1], axis=0)
    plt.figure(figsize=(8, 4))
    plt.plot(distances)
    plt.title('K-Distance Graph for DBSCAN')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{neighbors}-th nearest neighbor distance')
    plt.grid(True)
    plt.show()
    return distances[round(len(distances)*0.02)]  # Return value at 2% point

# 1. Hierarchical Clustering (with subsampling for dendrogram)
print("\n=== Hierarchical Clustering ===")
start = time.time()

# Full dataset for clustering
n_clusters = 3
hierarchical = AgglomerativeClustering(n_clusters=n_clusters,
                                     metric='euclidean',
                                     linkage='ward',
                                     memory='./cache')  # Cache for faster repeated runs
labels_hc = hierarchical.fit_predict(X_scaled)

# Subsample for dendrogram visualization
sample_idx = np.random.choice(len(X_scaled), size=100, replace=False)
X_sample = X_scaled[sample_idx]

plt.figure(figsize=(12, 6))
linkage_matrix = linkage(X_sample, method='ward')
dendrogram(linkage_matrix, truncate_mode='level', p=3)
plt.title("Wine Quality Hierarchical Clustering Dendrogram (Subsampled)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

plot_clusters(X_pca, labels_hc, "Wine Quality Hierarchical Clustering")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels_hc):.3f}")
print(f"Time taken: {time.time() - start:.2f} seconds")

# 2. Spectral Clustering
print("\n=== Spectral Clustering ===")
start = time.time()

spectral = SpectralClustering(n_clusters=n_clusters,
                            affinity='nearest_neighbors',
                            n_neighbors=20,
                            random_state=42)
labels_sc = spectral.fit_predict(X_scaled)

plot_clusters(X_pca, labels_sc, "Wine Quality Spectral Clustering")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels_sc):.3f}")
print(f"Time taken: {time.time() - start:.2f} seconds")

# 3. K-Means
print("\n=== K-Means Clustering ===")
start = time.time()

kmeans = KMeans(n_clusters=n_clusters,
               init='k-means++',
               n_init=10,
               random_state=42)
kmeans.fit(X_scaled)
labels_km = kmeans.labels_
centers = pca.transform(kmeans.cluster_centers_)  # Transform centers to PCA space

plot_clusters(X_pca, labels_km, "Wine Quality K-Means Clustering", centers)
print(f"Silhouette Score: {silhouette_score(X_scaled, labels_km):.3f}")
print(f"Time taken: {time.time() - start:.2f} seconds")

# 4. Gaussian Mixture Model (EM)
print("\n=== Gaussian Mixture Model ===")
start = time.time()

gmm = GaussianMixture(n_components=n_clusters,
                     covariance_type='diag',  # Faster than full covariance
                     random_state=42)
gmm.fit(X_scaled)
labels_gmm = gmm.predict(X_scaled)

plot_clusters(X_pca, labels_gmm, "Wine Quality GMM Clustering")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels_gmm):.3f}")
print(f"Time taken: {time.time() - start:.2f} seconds")

# 5. DBSCAN
print("\n=== DBSCAN ===")
start = time.time()

# Find optimal eps
optimal_eps = find_optimal_eps(X_scaled)
print(f"Optimal eps: {optimal_eps:.2f}")

dbscan = DBSCAN(eps=optimal_eps,
               min_samples=10,  # Increased for higher dimensions
               metric='euclidean')
labels_db = dbscan.fit_predict(X_scaled)

# Handle noise points (-1 labels) by assigning them to a special cluster
unique_labels = np.unique(labels_db)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

plot_clusters(X_pca, labels_db, f"Wine Quality DBSCAN (eps={optimal_eps:.2f}, {n_clusters} clusters)")
print(f"Number of clusters found: {n_clusters}")
print(f"Time taken: {time.time() - start:.2f} seconds")

# Compare all algorithms
plt.figure(figsize=(15, 10))
for i, (name, labels) in enumerate(zip(
    ['Hierarchical', 'Spectral', 'K-Means', 'GMM', 'DBSCAN'],
    [labels_hc, labels_sc, labels_km, labels_gmm, labels_db]), 1):
    plt.subplot(2, 3, i)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
    plt.title(name)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
plt.tight_layout()
plt.show()