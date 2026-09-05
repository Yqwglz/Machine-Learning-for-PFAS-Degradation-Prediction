# Machine-Learning-for-PFAS-Degradation-Prediction
This project applies machine learning to predict PFAS degradation behavior under various environmental and treatment conditions. The repository covers data preprocessing, feature engineering, data augmentation, model training/evaluation, and SHAP-based interpretability analysis.

本项目应用机器学习预测不同环境和处理条件下 PFAS 的降解行为。该代码库涵盖数据预处理、特征工程、数据增强、模型训练/评估以及基于 SHAP 的可解释性分析。

## Contents
```
PFAS-degradation/
├── data/
│   ├── PFAS.csv/                                 # Original dataset                                 原始数据集
│   ├── Original/                                 # The train/test dataset in the paper              论文训练/测试集
│   │   ├── test_Original_raw.csv                 # The Original test dataset(Unstandardized)        原始测试集（未标准化）
│   │   ├── test_Original_standardized.csv        # The Original test dataset(Standardized)          原始测试集（标准化）
│   │   ├── train_Original_raw.csv                # The Original train dataset(Unstandardized)       原始训练集（未标准化）
│   │   ├── train_Original_standardized.csv       # The Original train dataset(Standardized)         原始训练集（标准化）
│   ├── Splits/
│   │   ├── 859/
│   │       ├── test_859_raw.csv                 # The test dataset(Unstandardized)(random seed 859) 种子859划分下的测试集（未标准化）
│   │       ├── test_859_standardized.csv        # The test dataset(Standardized)(random seed 859)   种子859划分下的测试集（标准化）
│   │       ├── train_859_raw.csv                # The train dataset(Unstandardized)(random seed 859)种子859划分下的训练集（未标准化）
│   │       ├── train_859_standardized.csv       # The train dataset(Standardized)(random seed 859)  种子859划分下的训练集（标准化）                        
│   └── Augmented/                               # Augmented data in the paper                       论文增强数据 
├── task/
│   ├── preprocess.py                             # Data preprocess                                数据预处理
│   ├── Feature_lasso.py                          # Feature engineering                            特征工程
│   ├── VAEST.py                                  # Data augmentation                              数据增强
│   ├── Models.py                                 # Model training, evaluation and SHAP analysis   模型训练
├── results/
│   ├── Catboost/
│   ├── EN/
│   ├── MLP/
│   ├── ...            
├── requirements.txt                              #  The required packges                           依赖包
├── README.md
```
## Usage Instructions
```
If you wish to verify the results presented in our paper, you can directly use the standardized data in Data/Original for Lasso feature selection, then use Models to modify model parameters (Table S5, SM) and validate the benchmark results. Additionally, we provide the VAE-augmented dataset; you can use VAE_standardized and test_Original_standardized in the Augmented folder to modify model parameters and perform validation after VAE augmentation.


If you wish to verify our random seed, you can directly use Preprocess to perform dataset splitting and standardization, and compare with Data/859-0. The subsequent workflow is the same as described in the paper. After completing the benchmark model validation, use VAEST for data augmentation and then use Models again for model validation.


Note: You may modify the file save paths in each script as needed. All grid search parameters for all models are configured in Models and can be adjusted according to your requirements.
```
