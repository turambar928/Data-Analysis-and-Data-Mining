import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 设置中文显示和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn')

# 读取数据
df = pd.read_excel('anonymized data for workshop.xlsx')

# 数据预处理
def preprocess_data(df):
    # 选择相关指标
    metabolic_indicators = [
        '葡萄糖', '胰岛素', 'C肽1', '胰岛素（餐后2小时）', 'C肽（餐后2小时）',
        '促甲状腺素', '游离甲状腺素', '游离三碘甲状腺原氨酸',
        '钙', '磷', '镁', '钾', '钠', '氯',
        '甲状旁腺激素', '骨钙素(N-MID)', '碱性磷酸酶',
        '肌酐', '尿素', '尿酸'
    ]
    
    # 转换数值型列
    for col in metabolic_indicators:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, metabolic_indicators

# 1. 分析胰岛素信号通路
def analyze_insulin_pathway(df, indicators):
    # 选择胰岛素相关指标
    insulin_related = ['葡萄糖', '胰岛素', 'C肽1', '胰岛素（餐后2小时）', 'C肽（餐后2小时）']
    
    # 计算相关性矩阵
    insulin_corr = df[insulin_related].corr()
    
    # 可视化相关性
    plt.figure(figsize=(12, 10))
    sns.heatmap(insulin_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('胰岛素信号通路相关指标相关性分析', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('胰岛素通路相关性.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 分析胰岛素抵抗指数
    df['HOMA-IR'] = (df['葡萄糖'] * df['胰岛素']) / 22.5
    
    # 绘制HOMA-IR分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='HOMA-IR', bins=50)
    plt.axvline(x=2.5, color='r', linestyle='--', label='正常上限')
    plt.title('胰岛素抵抗指数(HOMA-IR)分布', fontsize=14)
    plt.xlabel('HOMA-IR')
    plt.ylabel('频数')
    plt.legend()
    plt.tight_layout()
    plt.savefig('HOMA-IR分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n胰岛素抵抗指数(HOMA-IR)统计：")
    print(df['HOMA-IR'].describe())

# 2. 分析甲状腺-骨代谢-钙磷平衡
def analyze_thyroid_bone_metabolism(df, indicators):
    # 选择相关指标
    bone_related = ['促甲状腺素', '游离甲状腺素', '钙', '磷', '甲状旁腺激素', '骨钙素(N-MID)', '碱性磷酸酶']
    
    # 计算相关性矩阵
    bone_corr = df[bone_related].corr()
    
    # 可视化相关性
    plt.figure(figsize=(14, 12))
    sns.heatmap(bone_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('甲状腺-骨代谢-钙磷平衡相关指标相关性分析', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('骨代谢相关性.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 分析钙磷乘积
    df['钙磷乘积'] = df['钙'] * df['磷']
    
    # 绘制钙磷乘积分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='钙磷乘积', bins=50)
    plt.axvline(x=2.5, color='g', linestyle='--', label='正常下限')
    plt.axvline(x=4.5, color='r', linestyle='--', label='正常上限')
    plt.title('钙磷乘积分布', fontsize=14)
    plt.xlabel('钙磷乘积')
    plt.ylabel('频数')
    plt.legend()
    plt.tight_layout()
    plt.savefig('钙磷乘积分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n钙磷乘积统计：")
    print(df['钙磷乘积'].describe())

# 3. 分析电解质与代谢异常
def analyze_electrolytes(df, indicators):
    # 选择电解质相关指标
    electrolyte_related = ['钾', '钠', '氯', '钙', '磷', '镁', '葡萄糖', '肌酐']
    
    # 计算相关性矩阵
    electrolyte_corr = df[electrolyte_related].corr()
    
    # 可视化相关性
    plt.figure(figsize=(14, 12))
    sns.heatmap(electrolyte_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('电解质与代谢指标相关性分析', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('电解质相关性.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 分析电解质异常与肾功能的关系
    kidney_related = ['钾', '钠', '氯', '肌酐', '尿素', '尿酸']
    kidney_corr = df[kidney_related].corr()
    
    # 可视化肾功能相关性
    plt.figure(figsize=(12, 10))
    sns.heatmap(kidney_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('电解质与肾功能指标相关性分析', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('肾功能相关性.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n电解质与肾功能指标相关性：")
    print(kidney_corr)

# 4. 进行聚类分析
def perform_clustering(df, indicators):
    # 准备数据
    data = df[indicators].dropna()
    
    if len(data) > 0:
        # 标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # PCA降维
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # 可视化聚类结果
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.title('代谢指标聚类分析', fontsize=14, pad=20)
        plt.xlabel('主成分1')
        plt.ylabel('主成分2')
        plt.colorbar(scatter, label='聚类')
        plt.tight_layout()
        plt.savefig('代谢聚类分析.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 分析各聚类的特征
        data['cluster'] = clusters
        cluster_stats = data.groupby('cluster').mean()
        print("\n各聚类中心特征：")
        print(cluster_stats)

# 执行分析
if __name__ == "__main__":
    print("开始分析代谢通路和离子通道关联...")
    df, indicators = preprocess_data(df)
    analyze_insulin_pathway(df, indicators)
    analyze_thyroid_bone_metabolism(df, indicators)
    analyze_electrolytes(df, indicators)
    perform_clustering(df, indicators)
    print("\n分析完成！请查看生成的图表文件。") 