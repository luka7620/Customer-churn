# 客户流失预测分析

## 项目描述

本项目旨在预测网络视频服务的客户流失情况。通过分析客户的各种特征（如订阅信息、使用情况等），构建预测模型来识别可能流失的客户，帮助企业采取相应措施提高客户留存率。

## 数据集介绍

数据集包含以下特征：

|特征名称 | 描述 | 中文释义 |
|--------|------|--------|
| id | unique subscriber id | 客户唯一ID |
| is_tv_subscriber | customer has a tv subscription? | 是否有电视订阅 |
| is_movie_package_subscriber | is he/she has a cinema movie package subs | 他/她有电影套餐订阅 |
| subscription_age | how many year has the customer use our service | 服务使用年限 |
| bill_avg | last 3 months bill avg | 过去3个月账单平均值 |
| remaining_contract | how many year remaining for customer contract. if null: customer hasn't have a contract, the customer who has a contract time have to use our service until contract end. if they canceled their service before contract time end they pay a penalty fare. | 客户合同剩余多少年。如果为空：客户没有合同，有合同的客户必须使用我们的服务直到合同结束。如果他们在合同期结束前取消服务，他们将支付违约金。 |
| service_failure_count | customer call count to call center for service failure for last 3 months | 过去3个月因服务失败致电客服中心的次数 |
| download_avg | last 3 months average usage (GB) | 过去3个月平均下载量(GB) |
| upload_avg | last 3 months upload avg (GB) | 过去3个月平均上传量(GB) |
| download_over_limit | most of customer has a download limit. if they reach this limit they have to pay for this. this column contain "most of last 3 months" | 大多数客户有下载限制。如果他们达到这个限制，他们必须为此付费。此列包含"过去3个月的大部分" |
| churn | whether the customer is churn | 客户是否流失 |

## 项目结构

```
├── data/
│   └── customer_churn.csv  # 客户流失数据集
├── results/                # 存放分析结果和图表
├── main.py                 # 主程序，集成分析结果
├── churn_prediction.py     # 主要预测模型脚本
├── churn_visualization.py  # 数据可视化脚本
├── ensemble_model.py       # 模型集成脚本
└── README.md               # 项目说明文档
```

## 脚本说明

### 1. main.py

主程序，提供完整的分析流程和结果汇总，包含以下功能：
- 数据加载和预处理
- 特征相关性分析
- 模型训练与评估比较
- 特征重要性分析
- 客户流失预防建议
- 分析结论和发现总结

### 2. churn_prediction.py

主要预测模型脚本，包含以下功能：
- 数据加载和探索性分析
- 数据预处理
- 特征工程和相关性分析
- 模型训练（逻辑回归、随机森林、梯度提升）
- 模型评估（准确率、精确率、召回率、F1值、ROC AUC）
- 特征重要性分析

### 3. churn_visualization.py

数据可视化脚本，包含以下功能：
- 目标变量分布可视化
- 数值特征分布分析
- 特征相关性热力图
- 箱线图分析
- ROC曲线比较
- 精确率-召回率曲线
- 特征重要性可视化
- 混淆矩阵可视化

### 4. ensemble_model.py

模型集成脚本，包含以下功能：
- 构建多个单一模型（逻辑回归、随机森林、梯度提升、SVM、KNN）
- 实现投票集成模型
- 实现堆叠集成模型
- 模型性能比较
- 集成模型可视化分析

## 使用方法

1. 确保安装了所需的Python库：
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. 运行主程序（快速获取分析结果）：
```
python main.py
```

3. 运行预测模型脚本（详细模型训练过程）：
```
python churn_prediction.py
```

4. 运行数据可视化脚本（生成可视化图表）：
```
python churn_visualization.py
```

5. 运行模型集成脚本（进行模型集成分析）：
```
python ensemble_model.py
```

6. 查看结果目录中的分析结果和图表。

## 分析结果

通过运行main.py主程序，可以获得以下核心分析结果：

1. 客户流失率及分布情况
2. 特征相关性分析及最重要特征识别
3. 三种模型（逻辑回归、随机森林、梯度提升）的性能比较
4. 特征重要性排序及解释
5. 基于分析的客户流失预防建议
6. 主要发现和结论总结

此外，通过运行其他脚本，可以获得更详细的分析图表和模型集成结果，这些结果可以帮助企业识别高流失风险的客户，并采取相应的客户留存策略。 