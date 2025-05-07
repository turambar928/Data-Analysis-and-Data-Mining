import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
# 设置支持中文字体，例如使用SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取并预处理数据
df = pd.read_excel("anonymized data for workshop.xlsx", parse_dates=['采集时间','入院时间','出院时间'])
df['Length_of_stay'] = (df['出院时间'] - df['入院时间']).dt.total_seconds() / (3600 * 24)
for col in ['钾','钠','氯','钙','磷','镁','肌酐','尿素','尿酸']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['葡萄糖']  = pd.to_numeric(df.get('葡萄糖',  np.nan), errors='coerce')
df['葡萄糖1'] = pd.to_numeric(df.get('葡萄糖1', np.nan), errors='coerce')
df['Glucose'] = df['葡萄糖'].combine_first(df['葡萄糖1'])
df['甲状旁腺激素']           = pd.to_numeric(df['甲状旁腺激素'],           errors='coerce')
df['甲状旁腺激素（pg/ml）']  = pd.to_numeric(df['甲状旁腺激素（pg/ml）'],  errors='coerce')
df['甲状旁腺激素（pg/ml）1'] = pd.to_numeric(df['甲状旁腺激素（pg/ml）1'], errors='coerce')
df['PTH_pg'] = df['甲状旁腺激素（pg/ml）'].combine_first(df['甲状旁腺激素（pg/ml）1'])
mask = df['PTH_pg'].isna() & df['甲状旁腺激素'].notna()
df.loc[mask, 'PTH_pg'] = df.loc[mask, '甲状旁腺激素'] * 9.43

# 2. 构建固定长度时间序列
time_counts = 10
def build_ts(df, col):
    lst = []
    for pid, grp in df.dropna(subset=[col]).groupby('住院号码'):
        vals = grp.sort_values('采集时间')[col].values
        if len(vals) < 2: continue
        seq = np.full(time_counts, np.nan)
        seq[:min(time_counts, len(vals))] = vals[:time_counts]
        lst.append([pid] + seq.tolist())
    return pd.DataFrame(lst, columns=['PID'] + [f'Time{t}' for t in range(time_counts)])

glucose_ts_df    = build_ts(df, 'Glucose')
creatinine_ts_df = build_ts(df, '肌酐')

# 3. 特征提取与聚类
def extract_cluster(ts_df):
    feats = []
    for _, row in ts_df.iterrows():
        pid = row['PID']
        seq = row[[f'Time{t}' for t in range(time_counts)]].dropna().values
        init, last = seq[0], seq[-1]
        pct = (last - init) / init if init != 0 else 0
        feats.append([pid, init, last, pct])
    feat_df = pd.DataFrame(feats, columns=['PID','init','last','pct']).dropna()
    X = StandardScaler().fit_transform(feat_df[['last','pct']])
    feat_df['cluster'] = KMeans(n_clusters=3, random_state=0).fit_predict(X)
    return feat_df

df_glu_feat = extract_cluster(glucose_ts_df)
df_cr_feat  = extract_cluster(creatinine_ts_df)
glucose_plot_df    = glucose_ts_df.merge(df_glu_feat[['PID','cluster']], on='PID')
creatinine_plot_df = creatinine_ts_df.merge(df_cr_feat [['PID','cluster']], on='PID')
# # 输出各聚类数目与均值
# counts = df_glu_feat['cluster_glu'].value_counts().sort_index()
# means = df_glu_feat.groupby('cluster_glu')[['last_glu','pct_change']].mean()
# print(counts, "\n", means)
# print(df_cr_feat.groupby('cluster_cr')[['last_cr','pct_change_cr']].mean())

# 4. 绘图函数
def plot_trajectories(df_plot, ylabel):
    clusters = sorted(df_plot['cluster'].unique())
    plt.figure(figsize=(12,4))
    for i, cl in enumerate(clusters,1):
        ax = plt.subplot(1,len(clusters),i)
        grp = df_plot[df_plot['cluster']==cl]
        n = min(3, len(grp))
        for _, r in grp.sample(n, random_state=42).iterrows():
            ax.plot(range(time_counts), r[[f'Time{t}' for t in range(time_counts)]], marker='o')
        ax.set_title(f'组{cl} (n={len(grp)})')
        ax.set_xticks(range(time_counts))
        ax.set_xlabel('随访次数')
        if i==1: ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def plot_box(df_plot, ylabel):
    ycol = f'Time{time_counts-1}'
    plot_data = df_plot[['cluster', ycol]].dropna()
    if plot_data.empty:
        print(f"⚠️ 无法绘制 {ylabel} 的箱线图：所有值均为 NaN")
        return
    plt.figure(figsize=(6,4))
    sns.boxplot(x='cluster', y=ycol, data=plot_data)
    plt.xlabel('聚类组')
    plt.ylabel(ylabel)
    plt.title(f'各组末次 {ylabel} 箱线图')
    plt.show()

def plot_pca(df_plot, ylabel):
    data = df_plot[[f'Time{t}' for t in range(time_counts)]].ffill(axis=1).fillna(0)
    coords = PCA(2).fit_transform(data)
    pdf = pd.DataFrame(coords, columns=['PC1','PC2'])
    pdf['cluster'] = df_plot['cluster']
    plt.figure(figsize=(6,5))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pdf, s=50)
    plt.title(f'{ylabel}轨迹聚类 PCA')
    plt.xlabel('主成分1'); plt.ylabel('主成分2')
    plt.legend(title='组别')
    plt.show()

# 5. 可视化
plot_trajectories(glucose_plot_df,    '血糖 (mmol/L)')
plot_pca(glucose_plot_df,             '血糖')

plot_trajectories(creatinine_plot_df, '肌酐 (mg/dL)')
plot_pca(creatinine_plot_df,          '肌酐')

# 6. 决策树规则挖掘
first_lab = df.sort_values(['住院号码','采集时间']).groupby('住院号码').first().reset_index()
patient   = first_lab[['住院号码','钙','磷','Glucose','肌酐','PTH_pg','Length_of_stay']]
patient['LongStay'] = (patient['Length_of_stay'] > patient['Length_of_stay'].median()).astype(int)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(patient[['钙','磷','Glucose','肌酐']].fillna(0), patient['LongStay'])
print("决策树规则：")
print(export_text(clf, feature_names=['钙','磷','Glucose','肌酐']))
