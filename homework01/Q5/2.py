import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis, euclidean, cityblock, cosine
#from sklearn.covariance import LedoitSchmidt
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import mahalanobis
'''
import seaborn as sns
'''
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


file_path = "air_quality_standardized.csv"
df = pd.read_csv(file_path)
df_city_avg = df.groupby('City').mean()
df_city_avg.to_csv("city_avg.csv")

df = pd.read_csv("city_avg.csv")


df_numeric = df.select_dtypes(include=[np.number])

pca = PCA(n_components=2)
pca.fit(df_numeric)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)


plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6, label="Individual Variance")
plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where="mid", linestyle='--', color='red', label="Cumulative Variance")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA - Explained Variance")
plt.legend()
plt.grid()
plt.show()
# 输出贡献率
for i, ratio in enumerate(explained_variance_ratio):
    print(f"主成分 {i+1}: 贡献率 = {ratio:.4f}")


# 设置显示所有列
pd.set_option('display.max_columns', None)

# 设置显示所有行（如果你只关心某一部分，可以设置具体行数）
pd.set_option('display.max_rows', None)

# 输出主成分载荷矩阵
components_df = pd.DataFrame(pca.components_, columns=df_numeric.columns, index=[f"PC{i+1}" for i in range(len(pca.components_))])
print("\n主成分载荷矩阵（每个变量的贡献）:")
print(components_df)




# 读取CSV文件
df = pd.read_csv('city_avg.csv')

# 提取数值数据
df_numeric = df.drop(columns=['City'])  # 删除城市名称列，只保留数值列

# 计算协方差矩阵
cov_matrix = np.cov(df_numeric.T)
inv_cov_matrix = np.linalg.inv(cov_matrix)

# 马氏距离计算函数
def mahalanobis_distance(x, y, inv_cov_matrix):
    return mahalanobis(x, y, inv_cov_matrix)

# 获取城市A, B, C的数值数据
city_A = df_numeric.iloc[0].values
city_B = df_numeric.iloc[1].values
city_C = df_numeric.iloc[2].values

# 计算城市A和城市B的马氏距离
mahalanobis_AB = mahalanobis_distance(city_A, city_B, inv_cov_matrix)
# 计算城市A和城市C的马氏距离
mahalanobis_AC = mahalanobis_distance(city_A, city_C, inv_cov_matrix)
# 计算城市B和城市C的马氏距离
mahalanobis_BC = mahalanobis_distance(city_B, city_C, inv_cov_matrix)

# 输出结果
print(f"City_A 和 City_B 的马氏距离：{mahalanobis_AB}")
print(f"City_A 和 City_C 的马氏距离：{mahalanobis_AC}")
print(f"City_B 和 City_C 的马氏距离：{mahalanobis_BC}")