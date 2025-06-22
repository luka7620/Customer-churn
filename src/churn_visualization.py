import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 加载数据
print("1. 加载数据")
df = pd.read_csv('data/customer_churn.csv', low_memory=False)

# 数据预处理
drop_cols = ['id', 'filter_$', 'ZRE_1', 'SRE_1', 'COO_1', 'LEV_1']
df = df.drop(drop_cols, axis=1)

# 检查列名拼写错误
if 'reamining_contract' in df.columns:
    df.rename(columns={'reamining_contract': 'remaining_contract'}, inplace=True)

# 处理数据类型问题
for col in ['download_avg', 'upload_avg']:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 填充缺失值
df = df.fillna(df.mean())

# 2. 数据可视化分析
print("\n2. 数据可视化分析")

# 创建结果目录
import os
if not os.path.exists('results'):
    os.makedirs('results')

# 2.1 目标变量分布
plt.figure(figsize=(8, 6))
sns.countplot(x='churn', data=df)
plt.title('客户流失分布')
plt.xlabel('流失状态 (0=未流失, 1=已流失)')
plt.ylabel('客户数量')
plt.savefig('results/churn_distribution.png')
plt.close()

# 2.2 数值特征分布
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('churn')  # 排除目标变量

plt.figure(figsize=(15, 12))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df, x=feature, hue='churn', bins=30, kde=True)
    plt.title(f'{feature} 分布')
plt.tight_layout()
plt.savefig('results/numeric_features_distribution.png')
plt.close()

# 2.3 特征相关性热力图
plt.figure(figsize=(12, 10))
correlation = df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
plt.title('特征相关性热力图')
plt.savefig('results/correlation_heatmap.png')
plt.close()

# 2.4 箱线图分析
plt.figure(figsize=(15, 12))
for i, feature in enumerate(numeric_features[:9], 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='churn', y=feature, data=df)
    plt.title(f'{feature} vs 流失状态')
plt.tight_layout()
plt.savefig('results/boxplot_analysis.png')
plt.close()

# 3. 模型训练与评估可视化
print("\n3. 模型训练与评估可视化")
# 准备特征和目标变量
X = df.drop('churn', axis=1)
y = df['churn']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3.1 训练模型
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 3.2 ROC曲线比较
plt.figure(figsize=(10, 8))
# 逻辑回归
lr_probs = lr_model.predict_proba(X_test_scaled)
y_prob_lr = lr_probs[:, 1]  # 使用索引1取正类概率
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, label=f'逻辑回归 (AUC = {roc_auc_lr:.3f})')

# 随机森林
rf_probs = rf_model.predict_proba(X_test_scaled)
# 手动提取正类概率
y_prob_rf = np.array([prob[1] for prob in rf_probs])
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC = {roc_auc_rf:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真阳性率 (TPR)')
plt.title('接收者操作特征曲线 (ROC)')
plt.legend(loc="lower right")
plt.savefig('results/roc_curve_comparison.png')
plt.close()

# 3.3 PR曲线比较
plt.figure(figsize=(10, 8))
# 逻辑回归
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_prob_lr)
plt.plot(recall_lr, precision_lr, label=f'逻辑回归')

# 随机森林
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
plt.plot(recall_rf, precision_rf, label=f'随机森林')

plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率曲线')
plt.legend(loc="best")
plt.savefig('results/pr_curve_comparison.png')
plt.close()

# 3.4 特征重要性可视化 (随机森林)
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf_model.feature_importances_
})

# 手动排序，不使用sort_values
importance_dict = dict(zip(feature_importance['特征'], feature_importance['重要性']))
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
sorted_features = [item[0] for item in sorted_importance]
sorted_values = [item[1] for item in sorted_importance]

plt.figure(figsize=(12, 8))
# 绘制条形图
plt.barh(range(len(sorted_features)), sorted_values, align='center')
plt.yticks(range(len(sorted_features)), sorted_features)
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.savefig('results/feature_importance.png')
plt.close()

# 3.5 混淆矩阵可视化
plt.figure(figsize=(12, 5))

# 逻辑回归
plt.subplot(1, 2, 1)
y_pred_lr = lr_model.predict(X_test_scaled)
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('逻辑回归混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')

# 随机森林
plt.subplot(1, 2, 2)
y_pred_rf = rf_model.predict(X_test_scaled)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('随机森林混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png')
plt.close()

print("\n可视化分析完成，结果保存在 'results' 目录中。") 