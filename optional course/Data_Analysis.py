import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# 设置中文显示（可选）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 Excel 文件
file_path = '塑料使用记录表（2025.4.16-5.14）.xlsx'
df = pd.read_excel(file_path)

# 预处理
df['日期'] = pd.to_datetime(df['日期'])
df.set_index('日期', inplace=True)

# 获取塑料类别列（排除“总计”和“备注”列）
plastic_types = df.columns.difference(['总计', '备注'])

# ---- 图1：各类塑料使用总量条形图 ----
total_usage = df[plastic_types].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=total_usage.values, y=total_usage.index, palette='Set2')
plt.title("各类塑料使用总量统计（2025.4.16 - 5.14）")
plt.xlabel("使用总次数")
plt.ylabel("塑料种类")
plt.tight_layout()
plt.show()

# ---- 图2：每天总使用趋势折线图 ----
plt.figure(figsize=(12, 4))
df['总计'].plot(marker='o')
plt.title("每日塑料使用总量趋势")
plt.ylabel("总计使用次数")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- 图3：各类塑料使用比例饼图（前5项）----
top5 = total_usage.head(5)
plt.figure(figsize=(6, 6))
plt.pie(top5, labels=top5.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Pastel1'))
plt.title("塑料使用最多的前5类占比")
plt.tight_layout()
plt.show()

# ---- 图4：每周使用总量堆叠柱状图 ----
df['周数'] = df.index.isocalendar().week
weekly_data = df.groupby('周数')[plastic_types].sum()

plt.figure(figsize=(12, 6))
weekly_data.plot(kind='bar', stacked=True, colormap='tab20')
plt.title("每周塑料使用情况堆叠图")
plt.ylabel("使用次数")
plt.xlabel("周数")
plt.xticks(rotation=0)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.show()
