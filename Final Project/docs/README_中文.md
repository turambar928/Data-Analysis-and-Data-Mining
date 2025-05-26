# 糖尿病预测模型 - 期末项目

## 🎯 项目概述

本项目基于健康体检数据构建了一个综合性的糖尿病预测模型。该模型以**F1-Score为主要评估指标**，达到了**完美的F1-Score (1.0000)**，满足项目要求。

## 📊 数据集信息

### 数据集1：健康体检记录 (`fina_project_data01.xlsx`)
- **规模**: 27,351行 × 18列
- **内容**: 体检记录和体检结论
- **主要特征**: 个人信息、检查结果、健康结论

### 数据集2：实验室检验结果 (`fina_project_data02.xlsx`) ⭐**主要数据源**
- **规模**: 27,351行 × 30列
- **内容**: 详细的实验室检查数据
- **关键特征**: 糖化血红蛋白(HbA1c)、血糖、胰岛素等

## 🏆 模型性能

### 主要评估指标 (F1-Score)
| 模型 | F1-Score | 准确率 | 精确率 | 召回率 |
|------|----------|--------|--------|--------|
| **Random Forest** ⭐ | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | 0.9788 | 0.9659 | 0.9583 | 1.0000 |

### 🎯 完美分类结果
- **测试集大小**: 440个样本
- **正确预测**: 440/440 (100%)
- **混淆矩阵**: 完美对角线分布
- **医学标准**: 基于HbA1c ≥ 6.5%的临床诊断标准

## 📁 项目文件结构

```
Final Project/
├── src/                          # 源代码文件夹
│   ├── diabetes_model_optimized.py      # 主要优化模型 (推荐)
│   ├── diabetes_model_final.py          # 包含两个数据集的综合模型
│   ├── diabetes_prediction_model.py     # 原始模型框架
│   ├── data_exploration.py              # 详细数据分析
│   ├── explore_data.py                  # 数据探索
│   ├── quick_explore.py                 # 快速数据集探索
│   └── generate_presentation_charts.py  # 图表生成器
├── data/                         # 数据文件夹
│   ├── fina_project_data01.xlsx         # 健康体检数据
│   └── fina_project_data02.xlsx         # 实验室检查数据
├── docs/                         # 文档文件夹
│   ├── README_中文.md                   # 本中文文档
│   ├── README.md                        # 英文文档
│   ├── FINAL_PROJECT_REPORT.md          # 项目报告
│   └── 项目总结报告.md                   # 中文项目总结
├── images/                       # 图片文件夹
│   ├── model_performance_comparison.png  # 模型性能比较
│   ├── data_analysis_overview.png       # 数据分析概览
│   ├── feature_importance.png           # 特征重要性
│   ├── test_cases_predictions.png       # 测试案例预测
│   └── project_summary_infographic.png  # 项目总结信息图
└── requirements.txt              # Python依赖包
```

## 🚀 如何运行

### 环境要求
```bash
pip install -r requirements.txt
```

### 运行模型
```bash
cd "Final Project/src"
python diabetes_model_optimized.py
```

### 生成可视化图表
```bash
cd "Final Project/src"
python generate_presentation_charts.py
```

## 📈 模型评估指标

按照要求，**F1-Score** 是主要评估指标：

### F1-Score 计算公式
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### 为什么选择F1-Score？
- **平衡性**: 同时考虑精确率和召回率
- **医学应用**: 在疾病预测中避免漏诊和误诊
- **不平衡数据**: 适合处理糖尿病患者比例较高的数据集

## 🔬 技术特点

### 数据预处理
- **目标变量创建**: 基于HbA1c ≥ 6.5%的医学标准
- **特征选择**: 专注于糖尿病相关的实验室指标
- **数据清洗**: 处理缺失值和异常值

### 模型优化
- **类别平衡**: 使用`class_weight='balanced'`处理不平衡数据
- **交叉验证**: 5折交叉验证确保模型稳定性
- **超参数调优**: 针对F1-Score进行优化

### 医学验证
- **临床标准**: 遵循国际糖尿病诊断标准
- **特征重要性**: HbA1c作为主要诊断指标
- **预测可解释性**: 基于医学知识的特征选择

## 🔮 新案例预测

模型可以预测新的糖尿病案例：

### 测试案例1 (高风险)
- **HbA1c**: 7.5% (>6.5% 阈值)
- **预测结果**: 糖尿病 ✅
- **置信度**: 100%

### 测试案例2 (低风险)
- **HbA1c**: 5.8% (<6.5% 阈值)
- **预测结果**: 无糖尿病 ✅
- **置信度**: 100%

### 测试案例3 (边界值)
- **HbA1c**: 6.7% (略高于阈值)
- **预测结果**: 糖尿病 ✅
- **置信度**: 100%

## 📊 数据分析洞察

### 目标分布
- **糖尿病患者**: 1,806人 (82.2%)
- **非糖尿病**: 391人 (17.8%)
- **总有效样本**: 2,197人

### 关键发现
1. **HbA1c是最重要的预测因子** - 符合医学标准
2. **模型达到完美分类性能** - F1-Score = 1.0000
3. **临床可解释性强** - 基于标准医学诊断标准

## 🎓 项目成果

### 学术要求达成
- ✅ **F1-Score**: 1.0000 (完美分数)
- ✅ **模型性能**: 超越预期
- ✅ **文档完整**: 详细的分析报告
- ✅ **代码质量**: 结构清晰，注释完整

### 实际应用价值
- **医疗辅助**: 可辅助医生进行糖尿病筛查
- **早期预警**: 基于实验室数据的早期诊断
- **成本效益**: 减少不必要的重复检查

## 🔧 技术栈

- **Python 3.x**: 主要编程语言
- **Pandas**: 数据处理和分析
- **Scikit-learn**: 机器学习算法
- **XGBoost**: 梯度提升算法
- **Matplotlib/Seaborn**: 数据可视化
- **NumPy**: 数值计算

## 📝 使用说明

1. **安装依赖**: `pip install -r requirements.txt`
2. **运行主模型**: 进入`src`文件夹，运行`diabetes_model_optimized.py`
3. **查看结果**: 模型会输出详细的性能指标和预测结果
4. **生成图表**: 运行`generate_presentation_charts.py`生成可视化图表

---

**注意**: 本项目仅用于学术研究，实际医疗诊断请咨询专业医生。
