import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# 假设已经加载了数据
df = pd.read_csv("medical_dataset.csv")

# 计算BloodSugar字段的均值和中位数
mean_bloodsugar = df["BloodSugar"].mean()
median_bloodsugar = df["BloodSugar"].median()

# 使用均值填充缺失值
df["BloodSugar_filled_mean"] = df["BloodSugar"].fillna(mean_bloodsugar)

# 使用中位数填充缺失值
df["BloodSugar_filled_median"] = df["BloodSugar"].fillna(median_bloodsugar)

# 使用KNN填充
knn_imputer = KNNImputer(n_neighbors=5)
df["BloodSugar_filled_knn"] = knn_imputer.fit_transform(df[["BloodSugar"]])

# 保存填充后的数据集为不同的CSV文件
df.to_csv("medical_dataset_filled.csv", index=False)


# 创建一个图表，用于展示填充前后的对比
plt.figure(figsize=(10, 6))

# 绘制填充前的数据
plt.hist(df["BloodSugar"].dropna(), bins=30, alpha=0.5, label="Original BloodSugar", color='blue')

# 绘制使用均值填充后的数据
plt.hist(df["BloodSugar_filled_mean"], bins=30, alpha=0.5, label="Mean Imputation", color='green')

# 绘制使用中位数填充后的数据
plt.hist(df["BloodSugar_filled_median"], bins=30, alpha=0.5, label="Median Imputation", color='orange')

# 绘制使用KNN填充后的数据
plt.hist(df["BloodSugar_filled_knn"], bins=30, alpha=0.5, label="KNN Imputation", color='red')

# 设置图表标题和标签
plt.title("Comparison of Imputation Methods for BloodSugar")
plt.xlabel("BloodSugar Value")
plt.ylabel("Frequency")
plt.legend()

# 显示图表
plt.show()
