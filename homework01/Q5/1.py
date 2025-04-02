import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv("air_quality_dataset.csv")

# === 1. 处理异常值（IQR 方法） ===
def remove_outliers_iqr(data):
    Q1 = data.quantile(0.25)  # 计算 Q1
    Q3 = data.quantile(0.75)  # 计算 Q3
    IQR = Q3 - Q1             # 计算 IQR
    lower_bound = Q1 - 1.5 * IQR  # 下界
    upper_bound = Q3 + 1.5 * IQR  # 上界
    return np.where((data < lower_bound) | (data > upper_bound), np.nan, data)

# 仅对数值型变量应用 IQR 方法
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = remove_outliers_iqr(df[col])

# === 2. 保存去除异常值后的数据 ===
df.to_csv("air_quality_no_outliers.csv", index=False)

print("✅ IQR 方法去除异常值完成，已保存至 air_quality_no_outliers.csv")

# 读取去除异常值后的数据
df = pd.read_csv("air_quality_no_outliers.csv")

# === 1. 仅对数值列进行 KNN 填充 ===
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 初始化 KNNImputer（使用 5 个最近邻）
imputer = KNNImputer(n_neighbors=5)

# 仅填充数值列
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# === 2. 保存填充后的数据 ===
df.to_csv("air_quality_filled.csv", index=False)

print("✅ KNN 填充完成，已保存至 air_quality_filled.csv")

# 读取 KNN 填充后的数据
df = pd.read_csv("air_quality_filled.csv")

# 初始化标准化器
scaler = StandardScaler()

# 仅对数值列进行标准化
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 保存标准化后的数据
df.to_csv("air_quality_standardized.csv", index=False)

print("✅ Z-score 标准化完成，已保存至 air_quality_standardized.csv")