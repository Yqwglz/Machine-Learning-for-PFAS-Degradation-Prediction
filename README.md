# Machine-Learning-for-PFAS-Degradation-Prediction
This project applies machine learning to predict PFAS degradation behavior under various environmental and treatment conditions. The repository covers data preprocessing, feature engineering, data augmentation, model training/evaluation, and SHAP-based interpretability analysis.

本项目应用机器学习预测不同环境和处理条件下 PFAS 的降解行为。该代码库涵盖数据预处理、特征工程、数据增强、模型训练/评估以及基于 SHAP 的可解释性分析。

## Contents
```
PFAS-degradation/
├── data/
│   ├── raw/                    # 原始数据
│   ├── processed/              # 处理后的数据
│   └── augmented/              # 增强后的数据
├── task/
│   ├── preprocess.py          # 数据预处理
│   ├── feature_engineering.py # 特征工程
│   ├── data_augmentation.py   # 数据增强
│   ├── train.py               # 模型训练
│   ├── evaluate.py            # 模型评估
│   └── shap_analysis.py       # SHAP 可解释性分析
├── results/
│   ├── figures/                # 结果图表
│   ├── metrics/                # 评估指标
│   └── shap_values/           # SHAP 值
├── configs/
│   └── config.yaml            # 配置文件
├── requirements.txt           # 依赖包
├── README.md
```
