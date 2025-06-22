# 客户流失预测分析系统

## 项目概述

本项目旨在通过机器学习技术预测网络视频服务的客户流失情况。通过分析客户的各种特征（如订阅信息、使用情况等），构建预测模型来识别可能流失的客户，帮助企业采取相应措施提高客户留存率。

## 数据集介绍

数据集包含以下关键特征：
- **id**: 客户唯一ID
- **is_tv_subscriber**: 是否有电视订阅
- **is_movie_package_subscriber**: 是否有电影套餐订阅
- **subscription_age**: 服务使用年限
- **bill_avg**: 过去3个月账单平均值
- **remaining_contract**: 客户合同剩余年限
- **service_failure_count**: 过去3个月因服务失败致电客服中心的次数
- **download_avg**: 过去3个月平均下载量(GB)
- **upload_avg**: 过去3个月平均上传量(GB)
- **download_over_limit**: 是否超过下载限制
- **churn**: 客户是否流失（目标变量）

## 项目结构

```
customer-churn/
│
├── data/                          # 数据文件夹
│   └── customer_churn.csv         # 客户流失数据集
│
├── src/                           # 源代码
│   ├── main.py                    # 主程序入口
│   ├── churn_prediction.py        # 流失预测模型实现
│   ├── churn_visualization.py     # 数据可视化模块
│   └── ensemble_model.py          # 集成模型实现
│
├── results/                       # 分析结果与可视化
│   ├── churn_distribution.png     # 流失分布图
│   ├── correlation_heatmap.png    # 特征相关性热图
│   ├── feature_importance.png     # 特征重要性分析
│   ├── model_performance_comparison.png  # 模型性能比较
│   └── roc_curve_comparison.png   # ROC曲线比较
│
├── .venv/                         # 虚拟环境
├── 客户流失预测分析.md              # 项目详细分析报告
└── README.md                      # 项目说明文档
```

## 安装指南

### 前置条件

- Python 3.8+
- pip 包管理器

### 安装步骤

1. 克隆项目仓库
```bash
git clone https://github.com/yourusername/customer-churn.git
cd customer-churn
```

2. 创建并激活虚拟环境
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. 安装依赖包
```bash
pip install -r requirements.txt
```

## 使用说明

### 数据预处理与探索

运行主程序进行数据预处理和探索性分析：

```bash
python src/main.py
```

### 模型训练与评估

训练并评估不同的预测模型：

```bash
python src/churn_prediction.py
```

### 集成模型构建

构建和评估集成模型：

```bash
python src/ensemble_model.py
```

### 数据可视化

生成数据可视化图表：

```bash
python src/churn_visualization.py
```

## 分析结果

### 模型性能

| 模型 | 准确率 | 精确率 | 召回率 | F1值 | ROC AUC |
|------|--------|--------|--------|------|---------|
| 随机森林 | 0.94 | 0.93 | 0.92 | 0.93 | 0.98 |
| 梯度提升 | 0.93 | 0.91 | 0.89 | 0.90 | 0.97 |
| 堆叠集成 | 0.94 | 0.93 | 0.91 | 0.92 | 0.98 |
| 投票集成 | 0.94 | 0.92 | 0.90 | 0.91 | 0.98 |
| 逻辑回归 | 0.87 | 0.84 | 0.85 | 0.84 | 0.94 |

随机森林和堆叠集成模型表现最佳，准确率约为94%，ROC AUC达到0.98。

### 特征重要性

分析表明以下特征对预测客户流失最为重要：
1. 合同剩余时间
2. 平均下载量
3. 平均上传量
4. 服务使用年限
5. 服务失败次数

### 业务建议

1. **合同管理策略**：对接近合同到期的客户提前采取挽留措施
2. **服务质量改进**：提升下载和上传服务质量，减少服务失败
3. **产品组合优化**：鼓励单一服务用户购买套餐组合
4. **使用行为激励**：鼓励客户更多地使用服务

## 贡献指南

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m '添加一些特性'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解更多详情 