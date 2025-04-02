import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

# 读取数据
df = pd.read_csv("medical_dataset.csv")

# 人为定义缺失值数量
# 假设缺失值为前50条 (BloodPressure缺失) 和前19条 (BloodSugar缺失)
df['BloodPressure_missing'] = 0
df['BloodSugar_missing'] = 0

# 人为设置缺失值
df.loc[:49, 'BloodPressure_missing'] = 1  # 假设前50条为缺失
df.loc[:18, 'BloodSugar_missing'] = 1  # 假设前19条为缺失

# MedicalHistory的缺失按比例设定，例如 15% 缺失
missing_count = int(len(df) * 0.15)
df['MedicalHistory_missing'] = 0
df.loc[:missing_count-1, 'MedicalHistory_missing'] = 1

# 记录所有字段的缺失情况
missing_flags = {
    "BloodPressure_missing": df["BloodPressure_missing"],
    "BloodSugar_missing": df["BloodSugar_missing"],
    "MedicalHistory_missing": df["MedicalHistory_missing"]
}
df_missing = pd.DataFrame(missing_flags)

# ---- 数值变量：使用 t-test 检测缺失是否与其他变量相关 ----
def test_mcar_numeric(variable, missing_flag):
    """ 进行 t-test 以检查数值型变量的缺失是否随机 """
    if variable not in df.columns or missing_flag not in df_missing.columns:
        print(f"变量 {variable} 不存在，跳过分析。")
        return

    complete = df[df_missing[missing_flag] == 0][variable].dropna()
    missing = df[df_missing[missing_flag] == 1][variable].dropna()

    if len(missing) == 0 or len(complete) == 0:
        print(f"⚠️ {variable} 的缺失数据过少（缺失 {len(missing)} 条，有效 {len(complete)} 条），无法进行 t-test。")
        return

    t_stat, p_value = ttest_ind(complete, missing, equal_var=False)
    print(f"{variable} 的 MCAR t-test p-value: {p_value:.5f}")
    if p_value > 0.05:
        print(f"✅ {variable} 可能是 MCAR（缺失随机分布）\n")
    else:
        print(f"⚠️ {variable} 可能是 MAR/MNAR（缺失与其他变量相关）\n")


# ---- 分类变量（文本）：使用 Chi-square 检测缺失是否与分类变量相关 ----
def test_mcar_categorical(categorical_var, missing_flag):
    """ 进行 Chi-Square 检验，检查分类变量的缺失是否随机 """
    if categorical_var not in df.columns or missing_flag not in df_missing.columns:
        print(f"变量 {categorical_var} 不存在，跳过分析。")
        return

    contingency_table = pd.crosstab(df[categorical_var], df_missing[missing_flag])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(f"{categorical_var} 的 MCAR Chi-Square p-value: {p:.5f}")
    if p > 0.05:
        print(f"✅ {categorical_var} 可能是 MCAR（缺失随机分布）\n")
    else:
        print(f"⚠️ {categorical_var} 可能是 MAR/MNAR（缺失与某些变量相关）\n")


# ---- 运行 MCAR 检测 ----
print("\n===== MCAR 检测：数值变量 =====")
test_mcar_numeric("BloodPressure", "BloodPressure_missing")
test_mcar_numeric("BloodSugar", "BloodSugar_missing")

print("\n===== MCAR 检测：分类变量（MedicalHistory） =====")
test_mcar_categorical("MedicalHistory", "MedicalHistory_missing")
