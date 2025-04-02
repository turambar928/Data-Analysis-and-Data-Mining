import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# 读取数据
df = pd.read_csv("air_quality_dataset.csv")

# 选择特征
features = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "Temp", "Humidity", "WindSpeed", "Pressure"]
data = df[features]


#原始数据直接 PCA + KMeans**
pca = PCA(n_components=2)
X_pca_raw = pca.fit_transform(data.dropna())  # 丢弃缺失值
kmeans_raw = KMeans(n_clusters=3, random_state=42).fit(X_pca_raw)
df_raw = pd.DataFrame(X_pca_raw, columns=["PC1", "PC2"])
df_raw["Cluster"] = kmeans_raw.labels_

#仅去除异常值后 PCA + KMeans**
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
filtered_data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
X_pca_filtered = pca.fit_transform(filtered_data.dropna())
kmeans_filtered = KMeans(n_clusters=3, random_state=42).fit(X_pca_filtered)
df_filtered = pd.DataFrame(X_pca_filtered, columns=["PC1", "PC2"])
df_filtered["Cluster"] = kmeans_filtered.labels_

#完整预处理后 PCA + KMeans**
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(filtered_data)  # KNN 填充
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)  # 归一化
X_pca_full = pca.fit_transform(data_scaled)
kmeans_full = KMeans(n_clusters=3, random_state=42).fit(X_pca_full)
df_full = pd.DataFrame(X_pca_full, columns=["PC1", "PC2"])
df_full["Cluster"] = kmeans_full.labels_

#可视化 PCA 降维后的聚类结果**
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(ax=axes[0], x="PC1", y="PC2", hue="Cluster", data=df_raw, palette="tab10")
axes[0].set_title("No Preprocessing")

sns.scatterplot(ax=axes[1], x="PC1", y="PC2", hue="Cluster", data=df_filtered, palette="tab10")
axes[1].set_title("Outlier Removed")

sns.scatterplot(ax=axes[2], x="PC1", y="PC2", hue="Cluster", data=df_full, palette="tab10")
axes[2].set_title("Full Preprocessing")

plt.show()
