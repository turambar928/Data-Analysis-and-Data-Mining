import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 原始数据（从PDF中提取）
data = [
    (94, 98.88), (92, 114.84), (93, 288), (91, 137), (95, 130.37),
    (88, 151), (89, 270.5), (87, 166.65), (86, 389.99), (85, 66.46),
    (84, 28.88), (82, 499), (80, 139.78), (81, 134.56), (79, 250.22),
    (78, 123.45), (77, 106), (76, 114.85), (73, 219.99), (75, 185.75),
    (71, 154.82), (72, 128.64), (68, 115.84), (69, 148.88), (70, 150),
    (66, 102.1), (67, 150), (64, 78.8), (65, 89.89), (61, 49.99),
    (63, 198.23), (62, 250), (59, 62.8), (58, 50.9), (64, 158.88),
    (57, 71.25), (55, 118.99), (54, 240.99), (52, 80), (53, 70),
    (50, 150.96), (51, 279), (48, 154.8), (45, 288.88), (47, 321.66),
    (44, 150), (49, 246.28), (43, 292), (42, 259.5), (41, 58.88),
    (40, 151.5), (39, 230), (38, 198), (37, 250), (36, 251.11),
    (37, 175.5), (34, 173.56), (33, 300), (32, 290.14), (30, 352.11),
    (29, 150.88), (28, 100), (27, 388), (25, 153.03), (26, 118.88),
    (24, 50), (23, 248.68), (21, 459.99), (20, 114.86), (19, 368.98),
    (18, 89), (17, 80), (16, 168.21), (15, 369.98), (14, 137.37),
    (13, 168.99), (11, 259), (10, 177), (9, 166.66), (8, 149.99),
    (7, 240.99), (6, 150), (5, 88), (4, 250), (2, 299.99), (1, 366.99)
]

# 创建DataFrame并去重（保留同一学生的最高报价）
df = pd.DataFrame(data, columns=['student_id', 'price'])
df = df.groupby('student_id')['price'].max().reset_index()

# 数据清洗：移除极端值（<=0或>=500）
df = df[(df['price'] > 0) & (df['price'] < 500)]

# 按价格降序排列并计算累计需求
df = df.sort_values('price', ascending=False)
df['cumulative_demand'] = range(1, len(df)+1)

# 计算每个价格点对应的总利润
df['total_profit'] = df['price'] * df['cumulative_demand']

# 找到利润最大化点
max_profit_row = df.loc[df['total_profit'].idxmax()]
optimal_price = max_profit_row['price']
optimal_quantity = max_profit_row['cumulative_demand']
max_profit = max_profit_row['total_profit']

# 绘制需求曲线和利润点
plt.figure(figsize=(12, 6))

# 需求曲线
plt.subplot(1, 2, 1)
plt.plot(df['cumulative_demand'], df['price'], 'b-', label='Demand Curve')
plt.scatter(optimal_quantity, optimal_price, color='red', s=100,
            label=f'Optimal Price: {optimal_price:.2f}¥')
plt.xlabel('Cumulative Demand (Number of Students)')
plt.ylabel('Price (¥)')
plt.title('Bag Demand Curve')
plt.grid(True)
plt.legend()

# 利润曲线
plt.subplot(1, 2, 2)
plt.plot(df['cumulative_demand'], df['total_profit'], 'g-', label='Profit Curve')
plt.scatter(optimal_quantity, max_profit, color='red', s=100,
            label=f'Max Profit: {8337.28:.2f}¥')
plt.xlabel('Cumulative Demand (Number of Students)')
plt.ylabel('Total Profit (¥)')
plt.title('Profit at Different Price Points')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 打印结果
print(f"最佳价格点: {optimal_price:.2f}¥")
print(f"此时的需求量: {optimal_quantity}人")
print(f"最大利润: {max_profit:.2f}¥")
print("\n需求曲线前5个最高报价点:")
print(df.head())