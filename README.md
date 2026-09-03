# Machine-Learning-for-PFAS-Degradation-Prediction
This project applies machine learning to predict PFAS degradation behavior under various environmental and treatment conditions. The repository covers data preprocessing, feature engineering, data augmentation, model training/evaluation, and SHAP-based interpretability analysis.

本项目应用机器学习预测不同环境和处理条件下 PFAS 的降解行为。该代码库涵盖数据预处理、特征工程、数据增强、模型训练/评估以及基于 SHAP 的可解释性分析。

## Contents
```
PFAS-degradation/
├── data/
│   ├── PFAS.csv/                                 # Original dataset                               原始数据集
│   ├── Original/                                 # The train/test dataset in the paper            论文训练/测试集
│   │   ├── test_Original_raw.csv                 # The Original test dataset(Unstandardized)      原始测试集（未标准化）
│   │   ├── test_Original_standardized.csv        # The Original test dataset(Standardized)        原始测试集（标准化）
│   │   ├── train_Original_raw.csv                # The Original train dataset(Unstandardized)     原始训练集（未标准化）
│   │   ├── train_Original_standardized.csv       # The Original train dataset(Standardized)       原始训练集（标准化）
│   ├── Augmented/                                # Augmented data in the paper                    论文增强数据
│   └── 
├── task/
│   ├── preprocess.py                             # Data preprocess                                数据预处理
│   ├── feature_lasso.py                          # Feature engineering                            特征工程
│   ├── VAE.py                                    # Data augmentation                              数据增强
│   ├── Models.py                                 # Model training, evaluation and SHAP analysis   模型训练
├── results/
│   ├── Catboost/
│   ├── EN/
│   ├── MLP/
│   ├── ...            
├── requirements.txt                              #  The required packges                           依赖包
├── README.md
```
