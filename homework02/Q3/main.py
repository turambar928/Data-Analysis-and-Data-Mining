import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

# 1. 数据准备 - 使用Raisin_3.xlsx数据
# 读取Excel文件
raisin_data = pd.read_excel('Raisin_3.xlsx')

# 假设数据集中最后一列是类别标签，其他列是特征
X = raisin_data.iloc[:, :-1].values  # 提取特征数据
y = raisin_data.iloc[:, -1].values  # 提取标签数据（如果有）

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 层次聚类 (使用Ward方法)
Z = linkage(X_scaled, method='ward')


# 3. 计算MSE的函数
def calculate_cluster_mse(points):
    """计算单个簇的均方误差(MSE)"""
    if len(points) == 0:
        return 0.0
    centroid = np.mean(points, axis=0)
    squared_errors = np.sum((points - centroid) ** 2, axis=1)
    return np.mean(squared_errors)


# 4. 跟踪合并过程中的MSE变化（改进版，保存簇信息）
def track_mse_changes_with_clusters(Z, X):
    """
    跟踪层次聚类每次合并时的MSE变化，并保存簇信息用于验证
    返回:
    - mse_records: 每次合并后的总MSE
    - delta_mse: 每次合并导致的MSE变化量
    - cluster_info: 每次合并时的簇信息（用于验证）
    """
    n_samples = len(X)
    mse_records = []
    delta_mse = []
    cluster_info = []

    # 初始化: 每个样本点自成一簇
    clusters = {i: {'points': [X[i]], 'size': 1, 'mse': 0.0, 'centroid': X[i]} for i in range(n_samples)}
    total_mse = 0.0
    mse_records.append(total_mse)


    print("合并前的MES：\n")
    for i in range(len(Z)):
        # 获取要合并的两个簇
        c1, c2 = int(Z[i, 0]), int(Z[i, 1])
        new_cluster_id = n_samples + i

        # 合并前的信息
        cluster1 = clusters[c1]
        cluster2 = clusters[c2]

        # 合并前的MSE
        mse_before = cluster1['mse'] * cluster1['size'] + cluster2['mse'] * cluster2['size']

        print(mse_before)

        # 合并后的新簇
        merged_points = cluster1['points'] + cluster2['points']
        merged_points_arr = np.array(merged_points)
        new_size = len(merged_points)
        new_centroid = np.mean(merged_points_arr, axis=0)
        new_mse = calculate_cluster_mse(merged_points_arr)

        # 计算MSE变化
        current_delta = new_mse * new_size - mse_before
        total_mse += current_delta

        # 记录结果
        mse_records.append(total_mse)
        delta_mse.append(current_delta)

        # 保存验证信息
        cluster_info.append({
            'step': i,
            'c1_points': cluster1['points'],
            'c2_points': cluster2['points'],
            'c1_centroid': cluster1['centroid'],
            'c2_centroid': cluster2['centroid'],
            'c1_size': cluster1['size'],
            'c2_size': cluster2['size'],
            'actual_delta': current_delta
        })

        # 更新簇字典
        clusters[new_cluster_id] = {
            'points': merged_points,
            'size': new_size,
            'mse': new_mse,
            'centroid': new_centroid
        }
        del clusters[c1], clusters[c2]

    return mse_records, delta_mse, cluster_info


# 5. 运行改进后的跟踪函数
mse_history, delta_mse_history, cluster_info = track_mse_changes_with_clusters(Z, X_scaled)

# 6. 可视化结果
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(mse_history)
plt.title('Total MSE During Clustering Process')
plt.xlabel('Merge Step')
plt.ylabel('Total MSE')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(delta_mse_history)
plt.title('ΔMSE at Each Merge Step')
plt.xlabel('Merge Step')
plt.ylabel('ΔMSE')
plt.grid(True)

# ΔMSE与合并距离的关系
merge_distances = Z[:, 2]
plt.subplot(2, 2, 3)
plt.scatter(merge_distances, delta_mse_history)
plt.title('Relationship Between Merge Distance and ΔMSE')
plt.xlabel('Merge Distance')
plt.ylabel('ΔMSE')
plt.grid(True)

# 簇大小对ΔMSE的影响
cluster_size_products = [info['c1_size'] * info['c2_size'] for info in cluster_info]
plt.subplot(2, 2, 4)
plt.scatter(cluster_size_products, delta_mse_history)
plt.title('Impact of Cluster Sizes on ΔMSE')
plt.xlabel('Product of Cluster Sizes (n1*n2)')
plt.ylabel('ΔMSE')
plt.grid(True)

plt.tight_layout()
plt.show()


# 7. 完整验证 ΔMSE = [n1*n2/(n1+n2)] * ||μ1-μ2||²
def theoretical_delta_mse(cluster1, cluster2):
    """理论计算ΔMSE"""
    n1, n2 = len(cluster1), len(cluster2)
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    distance_sq = np.sum((centroid1 - centroid2) ** 2)
    return (n1 * n2) / (n1 + n2) * distance_sq


print("\n验证最后5次合并的ΔMSE理论公式:")
for i in range(-5, 0):
    info = cluster_info[i]
    calculated_delta = theoretical_delta_mse(info['c1_points'], info['c2_points'])
    actual_delta = info['actual_delta']
    diff_percent = abs(calculated_delta - actual_delta) / actual_delta * 100

    print(f"\n合并步骤 {len(Z) + i}:")
    print(f"簇1大小: {info['c1_size']}, 簇2大小: {info['c2_size']}")
    print(f"实际ΔMSE: {actual_delta:.4f}")
    print(f"理论计算ΔMSE: {calculated_delta:.4f}")
    print(f"差异百分比: {diff_percent:.2f}%")
    print(f"理论公式验证: {'成功' if diff_percent < 1.0 else '失败'}")