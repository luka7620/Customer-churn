import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
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

# 2. 准备数据
print("\n2. 准备数据")
X = df.drop('churn', axis=1)
y = df['churn']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 定义评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n{model_name} 模型评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    
    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

# 4. 构建单一模型
print("\n3. 构建单一模型")
models = {
    'lr': LogisticRegression(max_iter=1000, random_state=42),
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'svm': SVC(probability=True, random_state=42),
    'knn': KNeighborsClassifier(n_neighbors=5)
}

model_results = {}
for name, model in models.items():
    model_results[name] = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)

# 5. 投票集成模型
print("\n4. 投票集成模型")
voting_clf = VotingClassifier(
    estimators=[
        ('lr', models['lr']),
        ('rf', models['rf']),
        ('gb', models['gb'])
    ],
    voting='soft'
)

voting_results = evaluate_model(voting_clf, X_train_scaled, X_test_scaled, y_train, y_test, "投票集成")
model_results['voting'] = voting_results

# 6. 堆叠集成模型
print("\n5. 堆叠集成模型")
base_models = [
    ('lr', models['lr']),
    ('rf', models['rf']),
    ('gb', models['gb']),
    ('knn', models['knn'])
]

# 堆叠集成 - 使用逻辑回归作为元学习器
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

stacking_results = evaluate_model(stacking_clf, X_train_scaled, X_test_scaled, y_train, y_test, "堆叠集成")
model_results['stacking'] = stacking_results

# 7. 模型比较
print("\n6. 模型比较")
models_comparison = pd.DataFrame({
    '模型': [result['name'] for result in model_results.values()],
    '准确率': [result['accuracy'] for result in model_results.values()],
    '精确率': [result['precision'] for result in model_results.values()],
    '召回率': [result['recall'] for result in model_results.values()],
    'F1值': [result['f1'] for result in model_results.values()],
    'ROC AUC': [result['auc'] for result in model_results.values()]
})

print(models_comparison.sort_values('ROC AUC', ascending=False))

# 8. 可视化模型比较
print("\n7. 可视化模型比较")
# 创建结果目录
import os
if not os.path.exists('results'):
    os.makedirs('results')

# 8.1 模型性能比较图
plt.figure(figsize=(12, 8))
metrics = ['准确率', '精确率', '召回率', 'F1值', 'ROC AUC']
for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    sns.barplot(x='模型', y=metric, data=models_comparison)
    plt.title(f'模型 {metric} 比较')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/model_performance_comparison.png')
plt.close()

# 8.2 ROC曲线比较
plt.figure(figsize=(12, 10))
for name, result in model_results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
    plt.plot(fpr, tpr, label=f"{result['name']} (AUC = {result['auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真阳性率 (TPR)')
plt.title('ROC曲线比较')
plt.legend(loc="lower right")
plt.savefig('results/ensemble_roc_comparison.png')
plt.close()

# 8.3 混淆矩阵比较 (最佳单一模型与最佳集成模型)
plt.figure(figsize=(15, 6))

# 找出最佳单一模型
single_models = {k: v for k, v in model_results.items() if k not in ['voting', 'stacking']}
best_single_model = max(single_models.values(), key=lambda x: x['auc'])

# 找出最佳集成模型
ensemble_models = {k: v for k, v in model_results.items() if k in ['voting', 'stacking']}
best_ensemble_model = max(ensemble_models.values(), key=lambda x: x['auc'])

# 绘制混淆矩阵
plt.subplot(1, 2, 1)
cm_single = confusion_matrix(y_test, best_single_model['y_pred'])
sns.heatmap(cm_single, annot=True, fmt='d', cmap='Blues')
plt.title(f'最佳单一模型: {best_single_model["name"]}')
plt.xlabel('预测标签')
plt.ylabel('真实标签')

plt.subplot(1, 2, 2)
cm_ensemble = confusion_matrix(y_test, best_ensemble_model['y_pred'])
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues')
plt.title(f'最佳集成模型: {best_ensemble_model["name"]}')
plt.xlabel('预测标签')
plt.ylabel('真实标签')

plt.tight_layout()
plt.savefig('results/best_models_confusion_matrix.png')
plt.close()

print(f"\n最佳单一模型: {best_single_model['name']}, ROC AUC: {best_single_model['auc']:.4f}")
print(f"最佳集成模型: {best_ensemble_model['name']}, ROC AUC: {best_ensemble_model['auc']:.4f}")
print("\n模型集成分析完成，结果保存在 'results' 目录中。") 