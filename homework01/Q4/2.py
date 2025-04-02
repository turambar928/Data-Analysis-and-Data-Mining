import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target

#数据标准化
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#使用PCA进行数据降维
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

#显示方差解释比例
print(f"Explained variance ratio (first two components): {pca.explained_variance_ratio_}")
print(f"Total variance explained by the first 2 components: {sum(pca.explained_variance_ratio_)}")

#可视化降维后的数据
plt.figure(figsize=(8, 6))
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.xlabel('Principal Components 1')
plt.ylabel('Principal Components 2')
plt.title('PCA of Iris Dataset')
plt.colorbar(label='Target Classes')
plt.show()