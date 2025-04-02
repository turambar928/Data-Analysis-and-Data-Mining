import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("ecommerce_dataset.csv")

# 1. 年龄 (age) 噪声检测
# 方法1：设定合理范围
age_outlier_mask = (df['age'] < 1) | (df['age'] > 150)

# 方法2：IQR 方法
Q1, Q3 = df['age'].quantile([0.25, 0.75])  # 计算25%和75%分位数
IQR = Q3 - Q1  # 计算四分位距
upper_bound = Q3 + 1.5 * IQR  # 计算上界
age_outlier_iqr_mask = df['age'] > max(upper_bound, 150)


# 最终年龄异常数据
age_outliers = df[age_outlier_mask | age_outlier_iqr_mask]

# 2. 购买金额 (purchase_amount) 噪声检测
# 方法1：设定合理范围
purchase_outlier_mask = df['purchase_amount'] > 999999

# 方法2：Z-Score 方法
mean_purchase = df['purchase_amount'].mean()
std_purchase = df['purchase_amount'].std()
purchase_outlier_z_mask = np.abs((df['purchase_amount'] - mean_purchase) / std_purchase) > 3

# 最终金额异常数据
purchase_outliers = df[purchase_outlier_mask | purchase_outlier_z_mask]

# 合并所有异常数据
removed_data = pd.concat([age_outliers, purchase_outliers]).drop_duplicates()

# 将清洗掉的异常数据保存到 removed_ecommerce_data.csv
removed_data.to_csv(r"removed_ecommerce_data.csv", index=False)

# 清洗后的数据
cleaned_df = df.drop(removed_data.index)

# 3. 年龄字段分箱
age_bins = [0, 17, 29, 44, 64, 150]
age_labels = ['Teen', 'Young Adult', 'Middle Aged', 'Older Adult', 'Senior']
cleaned_df['age_group'] = pd.cut(cleaned_df['age'], bins=age_bins, labels=age_labels, right=True)

# 处理异常年龄（>150），设为 Senior
cleaned_df.loc[cleaned_df['age'] > 150, 'age_group'] = 'Senior'

# 将清洗后的数据保存到 cleaned_ecommerce_dataset.csv
cleaned_df.to_csv(r"cleaned_ecommerce_dataset.csv", index=False)

print("数据清洗完成，清洗后的数据和被移除的数据已保存。")