import requests
import pandas as pd
import os

# 设置环境变量，禁用代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

def test_csv_prediction():
    """测试CSV文件批量预测功能"""
    
    print("🔍 测试XGBoost糖尿病CSV批量预测API")
    print("=" * 50)
    
    # 创建测试CSV数据
    test_data = {
        "age": [45, 50, 35, 60],
        "gender": ["Male", "Female", "Male", "Female"],
        "bmi": [28.5, 32.1, 24.3, 27.8],
        "HbA1c_level": [6.5, 7.2, 5.8, 6.9],
        "blood_glucose_level": [140, 160, 110, 150],
        "smoking_history": ["former", "never", "current", "never"],
        "hypertension": [1, 1, 0, 1],
        "heart_disease": [0, 1, 0, 0]
    }
    
    # 创建临时CSV文件
    df = pd.DataFrame(test_data)
    temp_csv = "test_diabetes_data.csv"
    df.to_csv(temp_csv, index=False)
    print(f"✅ 已创建测试CSV文件: {temp_csv}")
    
    try:
        # 1. 健康检查
        print("\n1. 健康检查...")
        response = requests.get(
            "http://1.94.9.72:5000/health",
            proxies={'http': None, 'https': None}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 服务状态: {data.get('status')}")
            print(f"✅ 模型已加载: {data.get('model_loaded')}")
        
        # 2. CSV批量预测
        print("\n2. CSV批量预测测试...")
        with open(temp_csv, 'rb') as f:
            files = {'file': (temp_csv, f, 'text/csv')}
            response = requests.post(
                "http://1.94.9.72:5000/predict_csv",
                files=files,
                proxies={'http': None, 'https': None}
            )
        
        # 打印预测结果
        if response.status_code == 200:
            result = response.json()
            print("\n预测结果:")
            print(f"总样本数: {result['total_samples']}")
            print(f"成功预测数: {result['successful_predictions']}")
            print(f"失败预测数: {result['failed_predictions']}")
            
            print("\n详细预测结果:")
            for pred in result['results']:
                print(f"行号 {pred['row_index']}:")
                print(f"  预测结果: {pred['prediction_text']}")
                print(f"  糖尿病概率: {pred['probability']['diabetes']:.2%}")
                print(f"  置信度: {pred['confidence']:.2%}")
                print("-" * 30)
        else:
            print(f"❌ 预测失败: {response.json()}")
    
    except Exception as e:
        print(f"❌ 测试过程中出错: {str(e)}")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            print(f"\n✅ 已清理测试CSV文件: {temp_csv}")

if __name__ == "__main__":
    test_csv_prediction() 