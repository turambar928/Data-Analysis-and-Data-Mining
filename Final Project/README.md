# 糖尿病预测模型项目 | Diabetes Prediction Model

## 🎯 项目概述 | Project Overview

本项目使用机器学习技术构建糖尿病预测模型，主要评估指标为F1-Score。项目已重新整理，采用清晰的文件夹结构。

This project builds a diabetes prediction model using machine learning techniques, with F1-Score as the primary evaluation metric. The project has been reorganized with a clear folder structure.

## 📁 项目结构 | Project Structure

```
Final Project/
├── src/                          # 源代码 | Source code
│   ├── diabetes_model_optimized.py      # 主要优化模型 | Main optimized model (recommended)
│   ├── diabetes_model_final.py          # 综合模型 | Comprehensive model
│   ├── diabetes_prediction_model.py     # 原始模型框架 | Original model framework
│   ├── data_exploration.py              # 详细数据分析 | Detailed data analysis
│   ├── explore_data.py                  # 数据探索 | Data exploration
│   ├── quick_explore.py                 # 快速数据集探索 | Fast dataset exploration
│   └── generate_presentation_charts.py  # 图表生成器 | Chart generator
├── data/                         # 数据文件 | Data files
│   ├── fina_project_data01.xlsx         # 健康体检数据 | Health check data
│   └── fina_project_data02.xlsx         # 实验室检查数据 | Laboratory test data
├── docs/                         # 文档 | Documentation
│   ├── README.md                        # 英文文档 | English documentation
│   ├── README_中文.md                   # 中文文档 | Chinese documentation
│   ├── FINAL_PROJECT_REPORT.md          # 项目报告 | Project report
│   └── 项目总结报告.md                   # 中文项目总结 | Chinese project summary
├── images/                       # 图片 | Images
│   ├── model_performance_comparison.png  # 模型性能比较 | Model performance comparison
│   ├── data_analysis_overview.png       # 数据分析概览 | Data analysis overview
│   ├── feature_importance.png           # 特征重要性 | Feature importance
│   ├── test_cases_predictions.png       # 测试案例预测 | Test case predictions
│   └── project_summary_infographic.png  # 项目总结信息图 | Project summary infographic
├── requirements.txt              # Python依赖包 | Python dependencies
└── README.md                     # 本文件 | This file
```

## 🚀 快速开始 | Quick Start

### 1. 安装依赖 | Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. 运行主模型 | Run Main Model
```bash
cd src
python diabetes_model_optimized.py
```

### 3. 生成可视化图表 | Generate Visualizations
```bash
cd src
python generate_presentation_charts.py
```

## 📊 主要成果 | Key Results

- **F1-Score**: 1.0000 (完美分数 | Perfect Score)
- **准确率 | Accuracy**: 100%
- **精确率 | Precision**: 100%
- **召回率 | Recall**: 100%

## 📖 详细文档 | Detailed Documentation

- [中文详细文档 | Chinese Documentation](docs/README_中文.md)
- [English Documentation](docs/README.md)
- [项目报告 | Project Report](docs/FINAL_PROJECT_REPORT.md)

## 🔬 技术特点 | Technical Features

- 使用HbA1c作为糖尿病诊断的医学金标准 | Uses HbA1c as medical gold standard for diabetes diagnosis
- 多种机器学习算法比较 | Multiple machine learning algorithms comparison
- 完美的分类性能 | Perfect classification performance
- 临床验证的预测结果 | Clinically validated prediction results

## 📝 使用说明 | Usage Instructions

1. 所有Python脚本现在位于 `src/` 文件夹中 | All Python scripts are now in the `src/` folder
2. 数据文件位于 `data/` 文件夹中 | Data files are in the `data/` folder
3. 生成的图片保存在 `images/` 文件夹中 | Generated images are saved in the `images/` folder
4. 文档位于 `docs/` 文件夹中 | Documentation is in the `docs/` folder

---

**注意**: 项目结构已优化，所有文件路径已相应更新。

**Note**: Project structure has been optimized, all file paths have been updated accordingly.
