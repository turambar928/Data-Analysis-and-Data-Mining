import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import os
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('whitegrid')  # seaborn 样式设置
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Create output directory for visualizations
os.makedirs('decision_support_results', exist_ok=True)

# Function to save figures
def save_fig(fig, filename):
    fig.savefig(os.path.join('decision_support_results', filename), bbox_inches='tight', dpi=300)
    plt.close(fig)

# English names for lab tests (for plotting)
lab_tests_en = {
    '钾': 'Potassium', 
    '钠': 'Sodium', 
    '氯': 'Chloride', 
    '钙': 'Calcium', 
    '磷': 'Phosphorus', 
    '镁': 'Magnesium', 
    '肌酐': 'Creatinine', 
    '尿素': 'Urea', 
    '尿酸': 'Uric Acid', 
    '糖化血红蛋白': 'HbA1c', 
    '葡萄糖': 'Glucose', 
    '葡萄糖(餐后2小时)': 'Glucose (2h postprandial)', 
    '游离甲状腺素': 'Free T4', 
    '总三碘甲状腺原氨酸': 'Total T3', 
    '低密度脂蛋白': 'LDL', 
    '甲状旁腺激素': 'PTH', 
    '总胆固醇': 'Total Cholesterol', 
    '甘油三酯': 'Triglycerides',
    '住院天数': 'Length of Stay',
    'HIS出院科室': 'Department',
    '骨钙素(N-MID)': 'Osteocalcin',
    '总I型胶原氨基端延长肽': 'Total Type I Collagen N-terminal Propeptide',
    '促甲状腺素': 'TSH',
    '促甲状腺素受体抗体': 'TSH Receptor Antibody',
    '甲状腺过氧化物酶抗体': 'Thyroid Peroxidase Antibody',
    '甲状腺球蛋白': 'Thyroglobulin',
    '甲状腺球蛋白抗体': 'Thyroglobulin Antibody',
    '总四碘甲状腺原氨酸': 'Total T4',
    '高密度脂蛋白': 'HDL',
    '25-羟基维生素D': 'Vitamin D (25-OH)',
    '碱性磷酸酶': 'Alkaline Phosphatase',
    '骨源碱性磷酸酶': 'Bone-specific Alkaline Phosphatase',
    '降钙素': 'Calcitonin',
    '尿钙': 'Urinary Calcium',
    '天门冬氨酸转氨酶': 'AST',
    'γ-谷氨酰转肽酶': 'GGT',
    '总胆红素': 'Total Bilirubin',
    '直接胆红素': 'Direct Bilirubin',
    '胰岛素': 'Insulin',
    'C肽': 'C-peptide',
    'β-胶原特殊序列': 'β-Collagen Special Sequence',
    '雌二醇': 'Estradiol',
    '促卵泡成熟素': 'FSH',
    '促黄体生成素': 'LH',
    '泌乳素': 'Prolactin',
    '孕酮': 'Progesterone',
    '睾酮': 'Testosterone',
    '硫酸去氢表雄酮': 'DHEA-S',
    '叶酸': 'Folic Acid',
    '维生素B12': 'Vitamin B12'
}

# Department translations
dept_mapping = {
    '内分泌代谢科': 'Endocrinology & Metabolism',
    '惠宾病房': 'VIP Ward',
    '日间病房': 'Day Ward'
}

print("Loading dataset...")
file_path = "anonymized data for workshop.xlsx"
df = pd.read_excel(file_path)
print(f"Dataset loaded with shape: {df.shape}")

# Convert date columns to datetime if not already
date_columns = ['入院时间', '出院时间', '采集时间']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# Calculate length of stay if not already done
if '住院天数' not in df.columns and '入院时间' in df.columns and '出院时间' in df.columns:
    df['住院天数'] = (df['出院时间'] - df['入院时间']).dt.total_seconds() / (24 * 3600)

# 1. Diagnostic Rule Extraction for Parathyroid Disorders
print("\n=== DIAGNOSTIC RULE EXTRACTION FOR PARATHYROID DISORDERS ===")

# Prepare data for parathyroid analysis
parathyroid_features = ['钙', '磷', '甲状旁腺激素', '甲状旁腺激素1', '甲状旁腺激素（pg/ml）', '甲状旁腺激素（pg/ml）1']
parathyroid_features = [f for f in parathyroid_features if f in df.columns]

if parathyroid_features:
    # Create a dataset for patients with parathyroid hormone measurements
    pth_df = df.dropna(subset=parathyroid_features, how='all').copy()
    
    # Convert to numeric
    for col in parathyroid_features:
        pth_df[col] = pd.to_numeric(pth_df[col], errors='coerce')
    
    # Define target variable: elevated PTH
    # Use any available PTH column
    pth_col = None
    for col in ['甲状旁腺激素', '甲状旁腺激素1', '甲状旁腺激素（pg/ml）', '甲状旁腺激素（pg/ml）1']:
        if col in pth_df.columns and pth_df[col].notna().sum() > 100:
            pth_col = col
            break
    
    if pth_col:
        print(f"Using {pth_col} as the PTH measurement")
        
        # Define normal ranges
        if '（pg/ml）' in pth_col:  # Different units
            pth_upper_limit = 65  # pg/ml
        else:
            pth_upper_limit = 6.9  # pmol/L
        
        # Create target variable
        pth_df['Elevated_PTH'] = (pth_df[pth_col] > pth_upper_limit).astype(int)
        
        # Select features for prediction
        features = ['钙', '磷']
        features = [f for f in features if f in pth_df.columns]
        
        if features and pth_df['Elevated_PTH'].sum() >= 10:  # Ensure we have enough positive cases
            # Prepare data
            X = pth_df[features].copy()
            y = pth_df['Elevated_PTH']
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed, y, test_size=0.3, random_state=42, stratify=y)
            
            # Train a decision tree classifier
            dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
            dt.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = dt.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\nDecision Tree Accuracy: {accuracy:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix for PTH Elevation Prediction')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            save_fig(plt.gcf(), 'pth_confusion_matrix.png')
            
            # Plot the decision tree
            plt.figure(figsize=(15, 10))
            plot_tree(dt, filled=True, feature_names=features, 
                     class_names=['Normal PTH', 'Elevated PTH'],
                     rounded=True, fontsize=10)
            plt.title('Decision Tree for Predicting Elevated PTH')
            save_fig(plt.gcf(), 'pth_decision_tree.png')
            
            # Extract and print the decision rules
            tree_rules = export_text(dt, feature_names=features)
            print("\nDecision Tree Rules:")
            print(tree_rules)
            
            # Save the rules to a file
            with open('decision_support_results/pth_decision_rules.txt', 'w') as f:
                f.write(tree_rules)
            
            # Create a more interpretable visualization of the rules
            # Plot calcium vs phosphorus with decision boundaries
            plt.figure(figsize=(10, 8))
            
            # Get min and max values for the plot
            x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
            
            # Create a meshgrid
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                np.arange(y_min, y_max, 0.01))
            
            # Predict for each point in the meshgrid
            Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot the decision boundary
            plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
            
            # Plot the data points
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolor='k', s=50)
            
            plt.xlabel(f'{lab_tests_en.get("钙", "Calcium")} (mmol/L)', fontsize=12)
            plt.ylabel(f'{lab_tests_en.get("磷", "Phosphorus")} (mmol/L)', fontsize=12)
            plt.title('Decision Boundary for PTH Elevation Prediction', fontsize=14)
            plt.colorbar(scatter, label='Elevated PTH')
            plt.grid(True, linestyle='--', alpha=0.7)
            save_fig(plt.gcf(), 'pth_decision_boundary.png')
            
            # Calculate feature importance
            importances = dt.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(X.shape[1]), importances[indices], align='center')
            plt.xticks(range(X.shape[1]), [features[i] for i in indices])
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Feature Importance for PTH Elevation Prediction')
            save_fig(plt.gcf(), 'pth_feature_importance.png')

# 2. Length of Stay Prediction
print("\n=== LENGTH OF STAY PREDICTION ===")

# Prepare data for length of stay prediction
if '住院天数' in df.columns:
    # Define long stay as > 7 days (adjust as needed)
    df['Long_Stay'] = (df['住院天数'] > 7).astype(int)
    
    # Select features for prediction
    los_features = ['钾', '钠', '氯', '钙', '磷', '肌酐', '尿素', '尿酸', '糖化血红蛋白', '葡萄糖', 'HIS出院科室']
    los_features = [f for f in los_features if f in df.columns]
    
    if los_features:
        # Create a dataset for LOS prediction
        los_df = df[los_features + ['Long_Stay']].copy()
        
        # Handle missing values and convert to numeric
        for col in los_features:
            if col != 'HIS出院科室':  # Skip categorical column
                los_df[col] = pd.to_numeric(los_df[col], errors='coerce')
        
        # Drop rows with missing target
        los_df = los_df.dropna(subset=['Long_Stay'])
        
        if len(los_df) > 100:  # Ensure we have enough data
            print(f"Preparing length of stay prediction model with {len(los_df)} patients")
            
            # Identify numeric and categorical features
            numeric_features = [f for f in los_features if f != 'HIS出院科室']
            categorical_features = ['HIS出院科室'] if 'HIS出院科室' in los_features else []
            
            # Create preprocessing pipeline
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            # Create and evaluate models
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }
            
            # Prepare data
            X = los_df[los_features]
            y = los_df['Long_Stay']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Train and evaluate each model
            results = {}
            
            for name, model in models.items():
                # Create pipeline with preprocessing
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                
                # Train the model
                pipeline.fit(X_train, y_train)
                
                # Evaluate the model
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store results
                results[name] = {
                    'accuracy': accuracy,
                    'pipeline': pipeline,
                    'y_pred': y_pred
                }
                
                print(f"\n{name} Accuracy: {accuracy:.3f}")
                print(classification_report(y_test, y_pred))
            
            # Compare model performance
            plt.figure(figsize=(10, 6))
            accuracies = [results[name]['accuracy'] for name in models.keys()]
            plt.bar(models.keys(), accuracies)
            plt.xlabel('Model')
            plt.ylabel('Accuracy')
            plt.title('Model Comparison for Length of Stay Prediction')
            plt.ylim(0.5, 1.0)  # Set y-axis to start from 0.5 for better visualization
            save_fig(plt.gcf(), 'los_model_comparison.png')
            
            # Select the best model
            best_model_name = max(results, key=lambda k: results[k]['accuracy'])
            best_pipeline = results[best_model_name]['pipeline']
            
            print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.3f}")
            
            # Plot confusion matrix for the best model
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {best_model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            save_fig(plt.gcf(), 'los_confusion_matrix.png')
            
            # Feature importance for the best model
            if best_model_name in ['Random Forest', 'Gradient Boosting']:
                # Extract the classifier from the pipeline
                classifier = best_pipeline.named_steps['classifier']
                
                # Get feature names after preprocessing
                if categorical_features:
                    # Get the categorical feature names after one-hot encoding
                    cat_encoder = best_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
                    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
                    feature_names = numeric_features + list(cat_feature_names)
                else:
                    feature_names = numeric_features
                
                # Get feature importances
                importances = classifier.feature_importances_
                
                # If the number of features after preprocessing doesn't match the importances length,
                # we'll just use indices instead of names
                if len(feature_names) != len(importances):
                    feature_names = [f'Feature {i}' for i in range(len(importances))]
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                # Plot feature importances
                plt.figure(figsize=(12, 8))
                plt.bar(range(len(importances)), importances[indices], align='center')
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.title(f'Feature Importance - {best_model_name}')
                plt.tight_layout()
                save_fig(plt.gcf(), 'los_feature_importance.png')
                
                # Print top 5 important features
                print("\nTop 5 important features for length of stay prediction:")
                for i in range(min(5, len(indices))):
                    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
            
            # Department-specific analysis
            if 'HIS出院科室' in los_features:
                # Calculate length of stay by department
                dept_los = df.groupby('HIS出院科室')['住院天数'].agg(['mean', 'median', 'std', 'count'])
                dept_los = dept_los.sort_values('mean', ascending=False)
                
                print("\nLength of stay by department:")
                print(dept_los)
                
                # Plot length of stay by department
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='HIS出院科室', y='住院天数', data=df)
                plt.title('Length of Stay by Department')
                plt.xlabel('Department')
                plt.ylabel('Length of Stay (days)')
                plt.xticks(rotation=45)
                save_fig(plt.gcf(), 'los_by_department.png')
                
                # Save department LOS statistics
                dept_los.to_csv('decision_support_results/los_by_department.csv')

# 3. Readmission Risk Prediction
print("\n=== READMISSION RISK PREDICTION ===")

# This would typically require readmission data which we don't have
# Instead, we'll create a proxy by identifying high-risk patients based on lab values

# Define high-risk criteria
high_risk_conditions = [
    (df['糖化血红蛋白'] > 9) & (df['肌酐'] > 130),  # High HbA1c and elevated creatinine
    (df['钾'] < 3.5) | (df['钾'] > 5.5),  # Abnormal potassium
    (df['钠'] < 135) | (df['钠'] > 145),  # Abnormal sodium
    (df['肌酐'] > 150)  # Severely elevated creatinine
]

# Convert to numeric first
for col in ['糖化血红蛋白', '肌酐', '钾', '钠']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create high risk flag
df['High_Risk'] = 0
for condition in high_risk_conditions:
    df['High_Risk'] = df['High_Risk'] | condition.fillna(False).astype(int)

# Count high risk patients
high_risk_count = df['High_Risk'].sum()
print(f"\nIdentified {high_risk_count} high-risk patients ({high_risk_count/len(df)*100:.1f}%)")

if high_risk_count > 10:  # Ensure we have enough high-risk patients
    # Analyze high risk patients by department
    if 'HIS出院科室' in df.columns:
        dept_risk = df.groupby('HIS出院科室')['High_Risk'].agg(['mean', 'sum', 'count'])
        dept_risk['percentage'] = dept_risk['sum'] / dept_risk['count'] * 100
        dept_risk = dept_risk.sort_values('percentage', ascending=False)
        
        print("\nHigh risk patients by department:")
        print(dept_risk)
        
        # Plot high risk percentage by department
        plt.figure(figsize=(10, 6))
        sns.barplot(x=dept_risk.index, y=dept_risk['percentage'])
        plt.title('Percentage of High Risk Patients by Department')
        plt.xlabel('Department')
        plt.ylabel('Percentage of High Risk Patients')
        plt.xticks(rotation=45)
        save_fig(plt.gcf(), 'high_risk_by_department.png')
        
        # Save department risk statistics
        dept_risk.to_csv('decision_support_results/risk_by_department.csv')
    
    # Build a predictive model for high risk patients
    risk_features = ['钾', '钠', '氯', '钙', '磷', '肌酐', '尿素', '尿酸', '糖化血红蛋白', '葡萄糖']
    risk_features = [f for f in risk_features if f in df.columns]
    
    if risk_features:
        # Create a dataset for risk prediction
        risk_df = df[risk_features + ['High_Risk']].copy()
        
        # Handle missing values and convert to numeric
        for col in risk_features:
            risk_df[col] = pd.to_numeric(risk_df[col], errors='coerce')
        
        # Drop rows with missing target
        risk_df = risk_df.dropna(subset=['High_Risk'])
        
        if len(risk_df) > 100:  # Ensure we have enough data
            print(f"\nPreparing high risk prediction model with {len(risk_df)} patients")
            
            # Create preprocessing pipeline
            preprocessor = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Prepare data
            X = risk_df[risk_features]
            y = risk_df['High_Risk']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Train a gradient boosting model
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(random_state=42))
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\nGradient Boosting Accuracy: {accuracy:.3f}")
            print(classification_report(y_test, y_pred))
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - High Risk Prediction')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            save_fig(plt.gcf(), 'high_risk_confusion_matrix.png')
            
            # Feature importance
            classifier = pipeline.named_steps['classifier']
            importances = classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [risk_features[i] for i in indices], rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Feature Importance - High Risk Prediction')
            plt.tight_layout()
            save_fig(plt.gcf(), 'high_risk_feature_importance.png')
            
            # Print top 5 important features
            print("\nTop 5 important features for high risk prediction:")
            for i in range(min(5, len(indices))):
                print(f"{risk_features[indices[i]]}: {importances[indices[i]]:.4f}")
            
            # Generate risk scores for all patients
            X_all = df[risk_features].copy()
            for col in risk_features:
                X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
            
            # Predict probabilities
            risk_probs = pipeline.predict_proba(X_all)[:, 1]
            
            # Add risk scores to the dataframe
            df['Risk_Score'] = np.nan
            df.loc[X_all.index, 'Risk_Score'] = risk_probs
            
            # Plot risk score distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df['Risk_Score'].dropna(), bins=20, kde=True)
            plt.title('Distribution of Risk Scores')
            plt.xlabel('Risk Score')
            plt.ylabel('Count')
            save_fig(plt.gcf(), 'risk_score_distribution.png')
            
            # Identify top 10% high risk patients
            high_risk_threshold = df['Risk_Score'].quantile(0.9)
            df['Very_High_Risk'] = (df['Risk_Score'] > high_risk_threshold).astype(int)
            
            very_high_risk_count = df['Very_High_Risk'].sum()
            print(f"\nIdentified {very_high_risk_count} very high-risk patients (top 10%)")
            
            # Save high risk patients to CSV
            high_risk_patients = df[df['Very_High_Risk'] == 1][['住院号码', 'Risk_Score'] + risk_features]
            high_risk_patients.to_csv('decision_support_results/very_high_risk_patients.csv', index=False)

print("\nClinical decision support analysis completed. Results saved to 'decision_support_results' folder.")
