import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# 1. 数据加载
print("1. 加载数据")
df = pd.read_csv('data/customer_churn.csv', low_memory=False)
print(f"数据形状: {df.shape}")

# 2. 数据探索
print("\n2. 数据探索")
print(f"数据列名: {df.columns.tolist()}")
print("\n数据类型:")
print(df.dtypes)
print("\n缺失值统计:")
print(df.isnull().sum())
print("\n目标变量分布:")
print(df['churn'].value_counts())
print(f"流失率: {df['churn'].mean()*100:.2f}%")

# 3. 数据预处理
print("\n3. 数据预处理")
# 删除不需要的列
drop_cols = ['id', 'filter_$', 'ZRE_1', 'SRE_1', 'COO_1', 'LEV_1']
df = df.drop(drop_cols, axis=1)

# 检查列名拼写错误
if 'reamining_contract' in df.columns:
    df.rename(columns={'reamining_contract': 'remaining_contract'}, inplace=True)

# 处理数据类型问题 - 更好的方式处理数值列
def clean_numeric_column(df, column):
    """清理数值列，移除非数字字符，转换为浮点数"""
    if df[column].dtype == 'object':
        try:
            # 尝试直接转换
            df[column] = pd.to_numeric(df[column], errors='coerce')
        except:
            print(f"无法直接转换列 {column}，尝试清理后转换")
    return df

# 处理特定的列
for col in ['download_avg', 'upload_avg']:
    df = clean_numeric_column(df, col)

# 填充缺失值
df = df.fillna(df.mean())

# 查看数据描述性统计
print("\n数据描述性统计:")
print(df.describe())

# 4. 特征工程
print("\n4. 特征工程")
# 相关性分析
print("特征相关性:")
corr = df.corr()
# 直接获取churn列的相关系数
churn_corr = corr['churn']
# 打印未排序的相关系数
print("原始相关系数：")
for col, val in churn_corr.items():
    print(f"{col}: {val:.4f}")

# 打印重要的相关系数
print("\n最重要的相关系数：")
important_features = ['customer_service_calls', 'total_charges', 'remaining_contract']
for feature in important_features:
    if feature in churn_corr:
        val = churn_corr[feature]
        print(f"{feature}: {val:.4f}")

# 5. 模型训练与评估
print("\n5. 模型训练与评估")
# 准备特征和目标变量
X = df.drop('churn', axis=1)
y = df['churn']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    # 手动提取正类概率
    y_prob = np.array([prob[1] for prob in probs])
    
    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy, auc

# 1. 逻辑回归模型
print("\n逻辑回归模型:")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model, lr_accuracy, lr_auc = evaluate_model(lr_model, X_train_scaled, X_test_scaled, y_train, y_test)

# 2. 随机森林模型
print("\n随机森林模型:")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model, rf_accuracy, rf_auc = evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_train, y_test)

# 3. 梯度提升模型
print("\n梯度提升模型:")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model, gb_accuracy, gb_auc = evaluate_model(gb_model, X_train_scaled, X_test_scaled, y_train, y_test)

# 模型比较
print("\n6. 模型比较")
models = ["逻辑回归", "随机森林", "梯度提升"]
accuracies = [lr_accuracy, rf_accuracy, gb_accuracy]
aucs = [lr_auc, rf_auc, gb_auc]

print("模型性能比较:")
for model, acc, auc in zip(models, accuracies, aucs):
    print(f"{model} - 准确率: {acc:.4f}, ROC AUC: {auc:.4f}")

# 特征重要性分析 (以随机森林模型为例)
print("\n7. 特征重要性分析 (随机森林)")
# 直接打印特征重要性
importances = rf_model.feature_importances_
features = X.columns
# 创建列表
importance_pairs = list(zip(features, importances))
# 手动实现排序 - 避免使用key函数
importance_pairs_sorted = []
for pair in importance_pairs:
    importance_pairs_sorted.append(pair)
# 手动冒泡排序
for i in range(len(importance_pairs_sorted)):
    for j in range(len(importance_pairs_sorted)-i-1):
        if importance_pairs_sorted[j][1] < importance_pairs_sorted[j+1][1]:
            importance_pairs_sorted[j], importance_pairs_sorted[j+1] = importance_pairs_sorted[j+1], importance_pairs_sorted[j]

# 输出排序后的特征重要性
for feature, importance in importance_pairs_sorted:
    print(f"{feature}: {importance:.4f}")

print("\n客户流失预测分析完成。") 