import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler


df = pd.read_excel("raisin_Dataset/Raisin_Dataset.xlsx")

#处理异常值
def remove_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return np.where((data < lower) | (data > upper), np.nan, data)

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = remove_outliers_iqr(df[col])

df.to_excel("Raisin_1.xlsx", index=False)
print("IQR方法去除异常值完成，保存至Raisin_1.xlsx")

df = pd.read_excel("Raisin_1.xlsx")
'''
对数值列进行KNN填充
'''
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
df.to_excel("Raisin_2.xlsx", index=False)
print("KNN填充完成，保存至Raisin_2.xlsx")

df = pd.read_excel("Raisin_2.xlsx")
'''
对数值列进行标准化处理
'''
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df.to_excel("Raisin_3.xlsx", index=False)
print("Z-score标准化完成，保存至Raisin_3.xlsx")

