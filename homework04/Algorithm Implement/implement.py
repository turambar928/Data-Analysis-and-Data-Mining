import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import weighted_mode
import matplotlib.pyplot as plt


class OverlapKMeans:
    def __init__(self, n_clusters, k_neighbors=10, gamma=None, max_iter=100, local=False, random_state=None):
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.gamma = gamma
        self.max_iter = max_iter
        self.local = local
        self.random_state = random_state
        np.random.seed(random_state)

    def fit(self, X):
        n_samples = X.shape[0]

        # Initialize centroids randomly
        centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        labels = self._assign_labels(X, centroids)

        # Find KNN for all points
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(X)
        knn_indices = nbrs.kneighbors(X, return_distance=False)

        # Calculate initial overlap values
        if self.local:
            overlap = self._calculate_overlap_v2(labels, X, knn_indices)
        else:
            overlap = self._calculate_overlap_v1(labels, X, knn_indices, centroids)

        if self.gamma is None:
            self.gamma_ = np.mean(overlap)
        else:
            self.gamma_ = self.gamma

        for _ in range(self.max_iter):
            old_labels = labels.copy()

            # Update centroids with overlap weights
            if self.local:
                labels = self._nn_partition(overlap, knn_indices, labels)
                overlap = self._calculate_overlap_v2(labels, X, knn_indices)
            else:
                centroids = self._update_centroids(labels, X, overlap)
                labels = self._assign_labels(X, centroids)
                overlap = self._calculate_overlap_v1(labels, X, knn_indices, centroids)

            if np.all(labels == old_labels):
                break

        self.labels_ = labels
        if not self.local:
            self.cluster_centers_ = centroids

        return self

    def _assign_labels(self, X, centroids):
        distances = pairwise_distances(X, centroids)
        return np.argmin(distances, axis=1)

    def _calculate_overlap_v1(self, labels, X, knn_indices, centroids):
        overlap = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            # Distance to own centroid (SSW term)
            d1 = np.linalg.norm(X[i] - centroids[labels[i]])

            # Find nearest neighbor in different cluster (SSB term)
            neighbors = knn_indices[i]
            different_cluster_mask = labels[neighbors] != labels[i]

            if np.any(different_cluster_mask):
                different_cluster_neighbors = neighbors[different_cluster_mask]
                distances = pairwise_distances(X[i].reshape(1, -1), X[different_cluster_neighbors])
                d2 = np.min(distances)
            else:
                # If no different cluster neighbors in KNN, find the closest different cluster point
                different_cluster_points = np.where(labels != labels[i])[0]
                if len(different_cluster_points) > 0:
                    distances = pairwise_distances(X[i].reshape(1, -1), X[different_cluster_points])
                    d2 = np.min(distances)
                else:
                    d2 = 1.0  # Only one cluster

            overlap[i] = d1 / d2 if d2 > 0 else 0

        return overlap

    def _calculate_overlap_v2(self, labels, X, knn_indices):
        overlap = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            # Find KNN in same cluster (local SSW term)
            neighbors = knn_indices[i]
            same_cluster_mask = labels[neighbors] == labels[i]

            if np.any(same_cluster_mask):
                same_cluster_neighbors = neighbors[same_cluster_mask]
                mean_shift = np.mean(pairwise_distances(X[i].reshape(1, -1), X[same_cluster_neighbors]))
            else:
                mean_shift = 0

            # Find nearest neighbor in different cluster (SSB term)
            different_cluster_mask = labels[neighbors] != labels[i]

            if np.any(different_cluster_mask):
                different_cluster_neighbors = neighbors[different_cluster_mask]
                distances = pairwise_distances(X[i].reshape(1, -1), X[different_cluster_neighbors])
                d2 = np.min(distances)
            else:
                # If no different cluster neighbors in KNN, find the closest different cluster point
                different_cluster_points = np.where(labels != labels[i])[0]
                if len(different_cluster_points) > 0:
                    distances = pairwise_distances(X[i].reshape(1, -1), X[different_cluster_points])
                    d2 = np.min(distances)
                else:
                    d2 = 1.0  # Only one cluster

            overlap[i] = mean_shift / d2 if d2 > 0 else 0

        return overlap

    def _update_centroids(self, labels, X, overlap):
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for j in range(self.n_clusters):
            cluster_mask = labels == j
            if np.sum(cluster_mask) == 0:
                centroids[j] = X[np.random.choice(X.shape[0], 1)]
                continue

            weights = np.exp(-(overlap[cluster_mask] / self.gamma_) ** 2)
            weighted_points = X[cluster_mask] * weights[:, np.newaxis]
            centroids[j] = np.sum(weighted_points, axis=0) / np.sum(weights)

        return centroids

    def _nn_partition(self, overlap, knn_indices, labels):
        new_labels = np.zeros_like(labels)

        for i in range(len(labels)):
            neighbors = knn_indices[i]
            weights = np.exp(-(overlap[neighbors] / self.gamma_) ** 2)
            # Use weighted mode to determine new label
            neighbor_labels = labels[neighbors]
            unique_labels = np.unique(neighbor_labels)
            weighted_counts = np.zeros(len(unique_labels))

            for k, label in enumerate(unique_labels):
                weighted_counts[k] = np.sum(weights[neighbor_labels == label])

            new_labels[i] = unique_labels[np.argmax(weighted_counts)]

        return new_labels


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Standard Overlap K-means
    okm = OverlapKMeans(n_clusters=4, k_neighbors=10, local=False)
    okm.fit(X)

    # Localized variant
    okm_local = OverlapKMeans(n_clusters=4, k_neighbors=10, local=True)
    okm_local.fit(X)

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=10)
    plt.title("True Clusters")

    plt.subplot(132)
    plt.scatter(X[:, 0], X[:, 1], c=okm.labels_, cmap='viridis', s=10)
    if hasattr(okm, 'cluster_centers_'):
        plt.scatter(okm.cluster_centers_[:, 0], okm.cluster_centers_[:, 1], c='red', marker='x')
    plt.title("Overlap K-means")

    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 1], c=okm_local.labels_, cmap='viridis', s=10)
    plt.title("Localized Overlap K-means")

    plt.tight_layout()
    plt.show()