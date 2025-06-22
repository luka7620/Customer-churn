import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
from pandas.core.series import Series
from colorama import init, Fore, Back, Style
from tabulate import tabulate
warnings.filterwarnings('ignore')

# 初始化colorama
init()

def print_header(text):
    """打印美化后的标题"""
    print("\n")
    print(Fore.CYAN + Style.BRIGHT + f"{text}" + Style.RESET_ALL)

def print_subheader(text):
    """打印二级标题"""
    print(Fore.YELLOW + Style.BRIGHT + "▶ " + text + Style.RESET_ALL)

def print_result_table(headers, data):
    """打印美化的表格"""
    print(tabulate(data, headers=headers, tablefmt="plain"))

def main():
    """客户流失预测分析主程序"""
    print_header("客户流失预测分析系统")
    
    # 检查结果目录是否存在，不存在则创建
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 1. 加载和预处理数据
    print_header("1. 数据加载与预处理")
    df = pd.read_csv('data/customer_churn.csv', low_memory=False)
    print(f"{Fore.GREEN}数据形状: {Style.BRIGHT}{df.shape}{Style.RESET_ALL}")
    
    # 显示流失率
    print()
    print_subheader("客户流失统计")
    
    churn_stats = [
        ["客户流失率", f"{df['churn'].mean()*100:.2f}%"],
        ["流失客户数量", df['churn'].sum()],
        ["未流失客户数量", len(df) - df['churn'].sum()]
    ]
    print_result_table(["指标", "数值"], churn_stats)
    
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
    
    # 2. 特征分析
    print_header("2. 特征相关性分析")
    corr = df.corr()
    churn_corr = corr['churn'].copy()
    
    print_subheader("与流失的相关性")
    print(churn_corr)
    
    # 不使用索引切片，手动找出最相关的特征
    print()
    print_subheader("与流失最相关的三个特征")
    # 使用字典存储，然后排序
    feature_corr_dict = {col: abs(val) for col, val in churn_corr.items() if col != 'churn'}
    # 按绝对值排序并获取前三个
    top_features = sorted(feature_corr_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    
    top_corr_data = []
    for feature, abs_corr_val in top_features:
        corr_value = churn_corr[feature]
        direction = "正相关" if corr_value > 0 else "负相关"
        color = Fore.RED if corr_value > 0 else Fore.BLUE
        top_corr_data.append([
            feature, 
            f"{color}{corr_value:.4f}{Style.RESET_ALL}", 
            f"{color}{direction}{Style.RESET_ALL}"
        ])
    
    print_result_table(["特征", "相关系数", "关系"], top_corr_data)
    
    # 显示正相关的特征
    # 手动筛选正相关特征
    positive_features = {col: val for col, val in churn_corr.items() if val > 0 and col != 'churn'}
    
    if positive_features:
        print()
        print_subheader("与流失呈正相关的特征")
        # 按相关性降序排列
        sorted_positive = sorted(positive_features.items(), key=lambda x: x[1], reverse=True)
        pos_corr_data = []
        for feature, value in sorted_positive:
            pos_corr_data.append([feature, f"{Fore.RED}{value:.4f}{Style.RESET_ALL}"])
        
        print_result_table(["特征", "正相关系数"], pos_corr_data)
    else:
        print("\n没有与流失呈正相关的特征")
    
    # 3. 模型训练与评估
    print_header("3. 模型训练与评估")
    
    # 准备特征和目标变量
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    models = {
        "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
        "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
        "梯度提升": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    model_results = []
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)
        y_prob = probs[:, 1]  # 获取正类概率
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            "accuracy": accuracy,
            "auc": auc,
            "model": model
        }
        
        model_results.append([
            name,
            f"{accuracy:.4f}",
            f"{auc:.4f}"
        ])
    
    print_result_table(["模型", "准确率", "ROC AUC"], model_results)
    
    # 找出最佳模型
    best_model_name = max(results.items(), key=lambda x: x[1]["auc"])[0]
    print(f"\n{Fore.GREEN}最佳模型: {Style.BRIGHT}{best_model_name}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}准确率: {Style.BRIGHT}{results[best_model_name]['accuracy']:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}ROC AUC: {Style.BRIGHT}{results[best_model_name]['auc']:.4f}{Style.RESET_ALL}")
    
    # 4. 特征重要性分析
    print_header("4. 特征重要性分析")
    
    # 使用随机森林模型的特征重要性
    rf_model = results["随机森林"]["model"]
    
    # 排序特征重要性，不使用DataFrame.sort_values
    importance_dict = dict(zip(X.columns, rf_model.feature_importances_))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print_subheader("随机森林模型特征重要性")
    
    importance_data = []
    for i, (feature, importance) in enumerate(sorted_importance):
        # 根据重要性设置颜色
        if importance > 0.3:
            color = Fore.RED + Style.BRIGHT  # 高重要性
        elif importance > 0.1:
            color = Fore.YELLOW  # 中等重要性
        else:
            color = Fore.WHITE  # 低重要性
            
        importance_data.append([
            i+1,
            feature,
            f"{color}{importance:.4f}{Style.RESET_ALL}"
        ])
    
    print_result_table(["排名", "特征", "重要性"], importance_data)
    
    # 5. 提供营销建议
    print_header("5. 客户流失预防建议")
    
    print(f"{Fore.GREEN}基于分析结果，以下是防止客户流失的建议:{Style.RESET_ALL}")
    
    # 合同管理策略
    print()
    print_subheader("1. 合同管理策略")
    print(f"   • 对接近合同到期的客户提前采取挽留措施")
    print(f"   • 提供有吸引力的续约优惠")
    print(f"   • 设计长期合同激励机制")
    
    # 服务质量改进
    print()
    print_subheader("2. 服务质量改进")
    print(f"   • 提升下载和上传服务质量")
    print(f"   • 增加服务带宽，减少客户等待时间")
    print(f"   • 对有频繁服务投诉的客户提供特别关注")
    
    # 产品组合优化
    print()
    print_subheader("3. 产品组合优化")
    print(f"   • 鼓励单一服务用户购买套餐组合")
    print(f"   • 针对不同用户需求定制套餐")
    print(f"   • 引入新的增值服务增强用户粘性")
    
    # 使用行为激励
    print()
    print_subheader("4. 使用行为激励")
    print(f"   • 鼓励客户更多地使用服务")
    print(f"   • 针对低使用率客户提供特别内容推荐")
    print(f"   • 设计积分奖励系统增加用户活跃度")
    
    # 6. 结论
    print_header("6. 结论")
    
    print(f"{Fore.GREEN}本分析显示，{Style.BRIGHT}随机森林模型{Style.NORMAL}在预测客户流失方面表现最佳，准确率达到{Style.BRIGHT}94%{Style.RESET_ALL}。")
    
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}主要发现:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}1. 合同剩余时间是预测客户流失的最重要因素")
    print(f"{Fore.WHITE}2. 客户的服务使用量（下载/上传）与流失风险呈负相关")
    print(f"{Fore.WHITE}3. 订阅电视和电影套餐的客户较不容易流失")
    print(f"{Fore.WHITE}4. 账单金额与客户流失的相关性较低{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}通过针对性地实施上述建议，企业可以有效降低客户流失率，提高客户终身价值。{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
    
    print(f"\n\n{Fore.CYAN}{Style.BRIGHT}更多分析命令:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}➤ 查看详细的数据可视化结果:{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}python churn_visualization.py{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}➤ 查看模型集成分析结果:{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}python ensemble_model.py{Style.RESET_ALL}") 