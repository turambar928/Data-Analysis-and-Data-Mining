import pandas as pd
from sklearn.mixture import GaussianMixture
from ucimlrepo import fetch_ucirepo
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



'''
# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
X = wine_quality.data.features
y = wine_quality.data.targets

# metadata
print(wine_quality.metadata)

# variable information
print(wine_quality.variables)
'''


data = pd.read_csv('winequality-white.csv', sep=';')
X = data.drop('quality', axis=1)
y = data['quality']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
'''
Hierarchical Clustering
'''
# 层次聚类
n_clusters = 3
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
labels_hc = hierarchical.fit_predict(X_scaled)

# 绘制树状图
plt.figure(figsize=(10, 6))
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix, truncate_mode='level', p=3)
plt.title(" Wine Quality Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_hc, cmap='viridis', edgecolor='k', s=50)
plt.title(" Wine Quality Hierarchical Clustering (Ward Linkage)")
plt.xlabel("Sepal Length (standardized)")
plt.ylabel("Sepal Width (standardized)")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

'''
spectral clustering
'''
# 谱聚类
spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=1.0, random_state=42)
labels_sc = spectral.fit_predict(X_scaled)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_sc, cmap='viridis', edgecolor='k', s=50)
plt.title("Wine Quality Spectral Clustering (RBF Kernel)")
plt.xlabel("Sepal Length (standardized)")
plt.ylabel("Sepal Width (standardized)")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


'''
KMeans
'''
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', edgecolor='k')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Wine Quality KMeans Clustering')
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.legend()
plt.show()

'''
Expectation Maximization
'''

n_clusters = 3
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(X_scaled)
labels = gmm.predict(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
plt.title('Wine Quality GMM (EM Algorithm) Clustering on Iris Dataset')
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Sepal Width (standardized)')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


'''
DBSCAN
'''
#使用DBSCAN聚类算法
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_pred = dbscan.fit_predict(X)

print("Wine Quality DBSCAN聚类结果：",y_pred)


