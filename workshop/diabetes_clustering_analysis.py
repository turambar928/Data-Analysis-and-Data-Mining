import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 设置字体以避免中文字符显示问题
plt.rcParams['font.family'] = 'SimHei'  # 使用 SimHei 字体，您可以根据实际字体进行修改
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 加载数据集
data = pd.read_excel('anonymized data for workshop.xlsx')

# 选择需要用于聚类的列
columns_to_use_for_clustering = ['胰岛素', '胰岛素（餐后2小时）', '葡萄糖', '葡萄糖(餐后2小时)', 'C肽1', '糖化血红蛋白', '糖化白蛋白']

# 清理数据：去除非数值字符，并转换为数值
def clean_and_convert_to_numeric(column):
    column = column.replace(r'[^\d.-]', '', regex=True)  # 替换非数字和负号字符
    return pd.to_numeric(column, errors='coerce')  # 转换为数值类型，无法转换的变成 NaN

# 对需要聚类的列进行数据清理
for col in columns_to_use_for_clustering:
    data[col] = clean_and_convert_to_numeric(data[col])

# 使用中位数填充缺失值，只填充我们要使用的数据
imputer = SimpleImputer(strategy='median')
data[columns_to_use_for_clustering] = imputer.fit_transform(data[columns_to_use_for_clustering])

# 检查填充后的数据
print("Check for missing values after imputation:")
print(data[columns_to_use_for_clustering].isnull().sum())  # 输出所选择列中的缺失值数量

# 确保填充后的数据没有缺失值
assert data[columns_to_use_for_clustering].isnull().sum().sum() == 0, "数据中仍然存在缺失值"

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[columns_to_use_for_clustering])

# 进行 KMeans 聚类
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)  # 显式设置 n_init
data['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# 可视化 KMeans 聚类结果（选择两个特征进行可视化）
plt.figure(figsize=(8, 6))
plt.scatter(data['胰岛素'], data['葡萄糖'], c=data['KMeans_Cluster'], cmap='viridis')
plt.xlabel('胰岛素')
plt.ylabel('葡萄糖')
plt.title('K-Means 聚类结果')
plt.show()

# 查看聚类分布
print("K-Means Cluster Distribution:")
print(data['KMeans_Cluster'].value_counts())

# 聚类结果的统计分析：每个簇的平均值
print("K-Means Cluster Statistics (Mean):")
print(data.groupby('KMeans_Cluster')[columns_to_use_for_clustering].mean())  # 按簇计算所选列的平均值
