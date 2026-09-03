import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import stats
import shap

warnings.filterwarnings('ignore')

# ============================================================================
# ===================== 配置区域（所有可调参数集中管理）=====================
# ============================================================================

# -------------------- 基础路径配置 --------------------
result_dir = "./result_Major revision/LOCO/mlp/990"
cv_dir = os.path.join(result_dir, "cv5")
shap_dir = os.path.join(result_dir, "shap_data")

# -------------------- 数据路径配置 --------------------
train_path = './Data_processed/LOGO/LOCO/train990.csv'
test_path = './Data_processed/LOGO/LOCO/test990.csv'

# 目标列名
target_column = "Kobs(h-1)"

# 要排除的列（不作为特征）
exclude_columns = ["Name", target_column]

# -------------------- 随机种子配置 --------------------
random_seed = 42

# -------------------- 交叉验证配置 --------------------
cv_n_splits = 5
cv_shuffle = True

# -------------------- MLP网格搜索参数 --------------------
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 100), (50, 30, 20), (100, 50, 25),],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],  # L2正则化参数
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.005, 0.01],
    'max_iter': [100, 500, 1000],
    'batch_size': [32, 64, 128],
    'early_stopping': [True],
}

# -------------------- MLP模型固定参数 --------------------
model_params = {
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 50
}

# -------------------- GridSearchCV配置 --------------------
grid_search_params = {
    'scoring': 'neg_mean_squared_error',
    'n_jobs': -1,
    'verbose': 1,
    'refit': True
}

# -------------------- SHAP分析配置 --------------------
shap_sample_size = 100
shap_max_samples = 10
shap_top_n_interactions = 10

# -------------------- 可视化配置 --------------------
font_sans_serif = ['Arial']
plot_style = 'seaborn-v0_8-darkgrid'
max_labels_display = 50
max_features_display = 50

# -------------------- 输出开关 --------------------
save_cv_data = True
save_shap_results = True
create_visualizations = True

# ============================================================================
# ===================== 配置区域结束 =========================================
# ============================================================================

# 设置中文字体
plt.rcParams['font.sans-serif'] = font_sans_serif
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(random_seed)


def ensure_directories():
    """确保所有必要的目录存在"""
    directories = [result_dir, cv_dir, shap_dir]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print(f"✓ 结果目录已创建: {result_dir}")
    return result_dir


def print_config():
    """打印当前配置信息"""
    print("\n" + "=" * 70)
    print("MLP 模型配置信息")
    print("=" * 70)
    print(f"基础结果目录: {result_dir}")
    print(f"训练数据路径: {train_path}")
    print(f"测试数据路径: {test_path}")
    print(f"目标列: {target_column}")
    print(f"随机种子: {random_seed}")
    print(f"交叉验证折数: {cv_n_splits}")
    print(f"SHAP样本数: {shap_sample_size}")
    print("\n网格搜索参数:")
    for key, value in param_grid.items():
        print(f"  {key}: {value}")
    print("=" * 70 + "\n")


def load_and_prepare_data():
    """加载并准备数据"""
    print("正在加载数据...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")

    # 清理列名空格
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    # 检查列名
    print("\n训练集列名:")
    print(train_df.columns.tolist())

    # 确定目标列
    target_col = target_column
    if target_col not in train_df.columns:
        possible_targets = [col for col in train_df.columns if "Kobs" in col]
        if possible_targets:
            target_col = possible_targets[0]
            print(f"使用列 '{target_col}' 作为目标变量")
        else:
            raise ValueError("未找到Kobs相关的目标列")

    # 使用配置中的排除列
    exclude_cols = list(exclude_columns)
    if target_col not in exclude_cols:
        exclude_cols.append(target_col)

    feature_columns = [col for col in train_df.columns if col not in exclude_cols]

    print(f"\n使用的特征数量: {len(feature_columns)}")
    print("特征列:", feature_columns)

    # 准备训练数据
    X_train = train_df[feature_columns].copy()
    y_train = train_df[target_col].copy()
    train_names = train_df["Name"].copy() if "Name" in train_df.columns else None

    # 准备测试数据
    X_test = test_df[feature_columns].copy()
    y_test = test_df[target_col].copy()
    test_names = test_df["Name"].copy() if "Name" in test_df.columns else None

    return (X_train, y_train, train_names,
            X_test, y_test, test_names,
            feature_columns, target_col)


def perform_grid_search(X_train_scaled, X_train_original, y_train, train_names=None,
                        feature_columns=None, target_column=None):
    """执行网格搜索进行超参数调优"""
    print("\n正在执行网格搜索...")

    # 创建KFold交叉验证
    kfold = KFold(n_splits=cv_n_splits, shuffle=cv_shuffle, random_state=random_seed)

    # ========== 导出各折数据 ==========
    if save_cv_data:
        print("\n正在导出交叉验证各折数据...")

        cv_dir_path = cv_dir
        os.makedirs(cv_dir_path, exist_ok=True)

        # 准备数据 - 使用原始数据
        X_df = pd.DataFrame(X_train_original, columns=feature_columns)
        y_series = pd.Series(y_train, name=target_column)

        # 确保Name列正确保存
        if train_names is not None:
            if isinstance(train_names, np.ndarray):
                name_series = pd.Series(train_names, name='Name')
            elif isinstance(train_names, pd.Series):
                name_series = train_names.reset_index(drop=True)
                name_series.name = 'Name'
            else:
                name_series = pd.Series(train_names, name='Name')
        else:
            name_series = None

        # 存储所有折的数据
        all_folds = {}

        # 遍历每一折
        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(X_train_scaled), 1):
            print(f"  第 {fold_idx} 折: 训练集 {len(train_indices)} 样本, 验证集 {len(val_indices)} 样本")

            # 训练集
            train_df = X_df.iloc[train_indices].copy()
            train_df[target_column] = y_series.iloc[train_indices].values
            if name_series is not None:
                train_df['Name'] = name_series.iloc[train_indices].values
            train_df['Fold'] = fold_idx
            train_df['Dataset_Type'] = 'Training'

            # 验证集
            val_df = X_df.iloc[val_indices].copy()
            val_df[target_column] = y_series.iloc[val_indices].values
            if name_series is not None:
                val_df['Name'] = name_series.iloc[val_indices].values
            val_df['Fold'] = fold_idx
            val_df['Dataset_Type'] = 'Validation'

            # 合并
            fold_df = pd.concat([train_df, val_df], ignore_index=True)

            # 重新排列列顺序，将Name放在第一列
            if name_series is not None:
                cols = ['Name'] + [col for col in fold_df.columns if col != 'Name']
                fold_df = fold_df[cols]

            all_folds[f'Fold_{fold_idx}'] = fold_df

        # 保存为CSV文件
        os.makedirs(cv_dir_path, exist_ok=True)
        for sheet_name, df in all_folds.items():
            df.to_csv(os.path.join(cv_dir_path, f"{sheet_name}.csv"), index=False)
        print(f"  ✓ CSV文件已保存到: {cv_dir_path}\n")

    # ========== 继续执行网格搜索 ==========
    # 创建MLP模型
    mlp = MLPRegressor(**model_params)

    # 创建GridSearchCV
    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=kfold,
        **grid_search_params
    )

    # 执行网格搜索（使用标准化数据）
    grid_search.fit(X_train_scaled, y_train)

    print("\n网格搜索完成!")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数 (负MSE): {grid_search.best_score_:.4f}")

    return grid_search


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """评估模型性能"""
    print("\n正在评估模型性能...")

    # 在各个数据集上进行预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算训练集指标
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)

    # 计算测试集指标
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 计算每个样本的平方误差（用于后续MSE贡献分析）
    train_squared_errors = (y_train - y_train_pred) ** 2
    test_squared_errors = (y_test - y_test_pred) ** 2

    # 计算每个样本对MSE的贡献占比
    train_total_mse = np.sum(train_squared_errors)
    test_total_mse = np.sum(test_squared_errors)

    train_mse_contributions = train_squared_errors / train_total_mse if train_total_mse > 0 else np.zeros_like(
        train_squared_errors)
    test_mse_contributions = test_squared_errors / test_total_mse if test_total_mse > 0 else np.zeros_like(
        test_squared_errors)

    # 计算过拟合程度
    overfitting_score = train_r2 - test_r2

    # 获取MLP网络信息
    n_layers = len(model.hidden_layer_sizes) + 1
    total_neurons = sum(model.hidden_layer_sizes) + 1

    # 计算网络参数数量（近似值）
    n_features = X_train.shape[1]
    n_neurons_hidden = model.hidden_layer_sizes[0] if isinstance(model.hidden_layer_sizes,
                                                                 tuple) else model.hidden_layer_sizes
    if isinstance(model.hidden_layer_sizes, tuple) and len(model.hidden_layer_sizes) > 1:
        n_params = (n_features * model.hidden_layer_sizes[0] + model.hidden_layer_sizes[0])
        for i in range(1, len(model.hidden_layer_sizes)):
            n_params += (model.hidden_layer_sizes[i - 1] * model.hidden_layer_sizes[i] + model.hidden_layer_sizes[i])
        n_params += (model.hidden_layer_sizes[-1] * 1 + 1)
    else:
        n_params = (n_features * n_neurons_hidden + n_neurons_hidden) + (n_neurons_hidden * 1 + 1)

    # 打印结果
    print("\n" + "=" * 70)
    print("MLP模型性能评估结果")
    print("=" * 70)
    print(f"{'数据集':<10} {'R²':<12} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'过拟合度':<12}")
    print(f"{'-' * 70}")
    print(f"{'训练集':<10} {train_r2:<12.4f} {train_mse:<12.4f} {train_rmse:<12.4f} {train_mae:<12.4f} {'-' * 12}")
    print(
        f"{'测试集':<10} {test_r2:<12.4f} {test_mse:<12.4f} {test_rmse:<12.4f} {test_mae:<12.4f} {overfitting_score:<12.4f}")
    print("=" * 70)
    print(f"\nMLP网络结构:")
    print(f"  隐藏层结构: {model.hidden_layer_sizes}")
    print(f"  总层数: {n_layers}")
    print(f"  总神经元数: {total_neurons}")
    print(f"  近似参数数量: {n_params:,}")
    print(f"  激活函数: {model.activation}")
    print(f"  优化器: {model.solver}")
    print(f"  过拟合程度: {overfitting_score:.4f}")

    return {
        'train': {
            'r2': train_r2,
            'mse': train_mse,
            'rmse': train_rmse,
            'mae': train_mae,
            'y_true': y_train,
            'y_pred': y_train_pred,
            'squared_errors': train_squared_errors,
            'mse_contributions': train_mse_contributions
        },
        'test': {
            'r2': test_r2,
            'mse': test_mse,
            'rmse': test_rmse,
            'mae': test_mae,
            'y_true': y_test,
            'y_pred': y_test_pred,
            'squared_errors': test_squared_errors,
            'mse_contributions': test_mse_contributions
        },
        'n_layers': n_layers,
        'total_neurons': total_neurons,
        'n_params': n_params,
        'hidden_layer_sizes': model.hidden_layer_sizes,
        'activation': model.activation,
        'overfitting_score': overfitting_score
    }


def save_results(model, results, feature_columns, target_column, train_names=None, test_names=None):
    """保存结果到文件"""
    print("\n正在保存结果...")

    result_dir_path = result_dir
    os.makedirs(result_dir_path, exist_ok=True)

    # 1. 保存模型参数和超参数
    params_path = os.path.join(result_dir_path, "model_parameters.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MLP Regression 模型参数\n")
        f.write("=" * 70 + "\n\n")

        f.write("网络结构:\n")
        f.write(f"  隐藏层结构: {model.hidden_layer_sizes}\n")
        f.write(f"  总层数: {results['n_layers']}\n")
        f.write(f"  总神经元数: {results['total_neurons']}\n")
        f.write(f"  近似参数数量: {results['n_params']:,}\n\n")

        f.write("超参数:\n")
        f.write(f"  激活函数: {model.activation}\n")
        f.write(f"  优化器 (solver): {model.solver}\n")
        f.write(f"  L2正则化 (alpha): {model.alpha}\n")
        f.write(f"  学习率: {model.learning_rate}\n")
        f.write(f"  初始学习率: {model.learning_rate_init}\n")
        f.write(f"  最大迭代次数: {model.max_iter}\n")
        f.write(f"  批次大小: {model.batch_size}\n")
        f.write(f"  动量 (momentum): {model.momentum if hasattr(model, 'momentum') else 'N/A'}\n")
        f.write(f"  早停: {model.early_stopping}\n")
        f.write(f"  随机种子: {model.random_state}\n\n")

        f.write("训练历史:\n")
        f.write(f"  最终损失值: {model.loss_:.6f}\n")
        f.write(f"  迭代次数: {model.n_iter_}\n")
        f.write(f"  输出层激活函数: {model.out_activation_}\n\n")

        f.write(f"模型性能总结:\n")
        f.write(f"  训练集 R²: {results['train']['r2']:.4f}\n")
        f.write(f"  测试集 R²: {results['test']['r2']:.4f}\n")
        f.write(f"  过拟合程度: {results['overfitting_score']:.4f}\n")

        f.write(f"\nMSE贡献分析:\n")
        f.write(f"  训练集总MSE: {results['train']['mse'] * len(results['train']['y_true']):.4f}\n")
        f.write(f"  测试集总MSE: {results['test']['mse'] * len(results['test']['y_true']):.4f}\n")
        f.write(
            f"  训练集样本MSE贡献范围: {results['train']['mse_contributions'].min():.6f} - {results['train']['mse_contributions'].max():.6f}\n")
        f.write(
            f"  测试集样本MSE贡献范围: {results['test']['mse_contributions'].min():.6f} - {results['test']['mse_contributions'].max():.6f}\n")

    print(f"模型参数已保存到: {params_path}")

    # 2. 保存性能指标
    metrics_path = os.path.join(result_dir_path, "model_performance.csv")
    metrics_df = pd.DataFrame({
        'Dataset': ['Train', 'Test'],
        'R²': [results['train']['r2'], results['test']['r2']],
        'MSE': [results['train']['mse'], results['test']['mse']],
        'RMSE': [results['train']['rmse'], results['test']['rmse']],
        'MAE': [results['train']['mae'], results['test']['mae']],
        'Total_Squared_Errors': [
            np.sum(results['train']['squared_errors']),
            np.sum(results['test']['squared_errors'])
        ],
        'Overfitting_Score': ['-', results['overfitting_score']],
        'Hidden_Layers': [str(results['hidden_layer_sizes'])] * 2,
        'Activation': [results['activation']] * 2,
        'Total_Neurons': [results['total_neurons']] * 2,
        'Total_Params': [results['n_params']] * 2
    })
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8')
    print(f"性能指标已保存到: {metrics_path}")

    # 3. 保存预测结果
    # 准备训练集数据
    train_data = {
        'Dataset': ['train'] * len(results['train']['y_true']),
        'Sample_Index': list(range(1, len(results['train']['y_true']) + 1)),
        'True_Value': results['train']['y_true'],
        'Predicted_Value': results['train']['y_pred'],
        'Residual': results['train']['y_true'] - results['train']['y_pred'],
        'Absolute_Error': np.abs(results['train']['y_true'] - results['train']['y_pred']),
        'Squared_Error': results['train']['squared_errors'],
        'MSE_Contribution': results['train']['mse_contributions'],
        'MSE_Contribution_Percent': results['train']['mse_contributions'] * 100
    }

    # 准备测试集数据
    test_data = {
        'Dataset': ['test'] * len(results['test']['y_true']),
        'Sample_Index': list(range(1, len(results['test']['y_true']) + 1)),
        'True_Value': results['test']['y_true'],
        'Predicted_Value': results['test']['y_pred'],
        'Residual': results['test']['y_true'] - results['test']['y_pred'],
        'Absolute_Error': np.abs(results['test']['y_true'] - results['test']['y_pred']),
        'Squared_Error': results['test']['squared_errors'],
        'MSE_Contribution': results['test']['mse_contributions'],
        'MSE_Contribution_Percent': results['test']['mse_contributions'] * 100
    }

    # 添加样本名称
    if train_names is not None:
        if isinstance(train_names, np.ndarray):
            train_data['Sample_Name'] = train_names.tolist()
        elif isinstance(train_names, pd.Series):
            train_data['Sample_Name'] = train_names.tolist()
        else:
            train_data['Sample_Name'] = list(train_names)

    if test_names is not None:
        if isinstance(test_names, np.ndarray):
            test_data['Sample_Name'] = test_names.tolist()
        elif isinstance(test_names, pd.Series):
            test_data['Sample_Name'] = test_names.tolist()
        else:
            test_data['Sample_Name'] = list(test_names)

    # 合并训练和测试数据
    predictions_df = pd.DataFrame({
        'Dataset': train_data['Dataset'] + test_data['Dataset'],
        'Sample_Index': train_data['Sample_Index'] + test_data['Sample_Index'],
        'True_Value': np.concatenate([train_data['True_Value'], test_data['True_Value']]),
        'Predicted_Value': np.concatenate([train_data['Predicted_Value'], test_data['Predicted_Value']]),
        'Residual': np.concatenate([train_data['Residual'], test_data['Residual']]),
        'Absolute_Error': np.concatenate([train_data['Absolute_Error'], test_data['Absolute_Error']]),
        'Squared_Error': np.concatenate([train_data['Squared_Error'], test_data['Squared_Error']]),
        'MSE_Contribution': np.concatenate([train_data['MSE_Contribution'], test_data['MSE_Contribution']]),
        'MSE_Contribution_Percent': np.concatenate(
            [train_data['MSE_Contribution_Percent'], test_data['MSE_Contribution_Percent']])
    })

    if train_names is not None or test_names is not None:
        sample_names = []
        if train_names is not None:
            if isinstance(train_names, np.ndarray):
                sample_names.extend(train_names.tolist())
            elif isinstance(train_names, pd.Series):
                sample_names.extend(train_names.tolist())
            else:
                sample_names.extend(list(train_names))
        else:
            sample_names.extend([f'Unknown_train_{i}' for i in range(len(train_data['True_Value']))])
        if test_names is not None:
            if isinstance(test_names, np.ndarray):
                sample_names.extend(test_names.tolist())
            elif isinstance(test_names, pd.Series):
                sample_names.extend(test_names.tolist())
            else:
                sample_names.extend(list(test_names))
        else:
            sample_names.extend([f'Unknown_test_{i}' for i in range(len(test_data['True_Value']))])
        predictions_df.insert(1, 'Sample_Name', sample_names)

    # 按贡献百分比排序（降序）
    predictions_df = predictions_df.sort_values('MSE_Contribution_Percent', ascending=False)
    predictions_df['Rank_by_MSE_Contribution'] = range(1, len(predictions_df) + 1)

    # 保存预测结果
    pred_path = os.path.join(result_dir_path, "predictions.csv")
    predictions_df.to_csv(pred_path, index=False, encoding='utf-8')
    print(f"预测结果（含MSE贡献分析）已保存到: {pred_path}")

    # 4. 保存损失曲线
    if hasattr(model, 'loss_curve_'):
        loss_df = pd.DataFrame({
            'Epoch': range(1, len(model.loss_curve_) + 1),
            'Loss': model.loss_curve_
        })
        loss_path = os.path.join(result_dir_path, "training_loss.csv")
        loss_df.to_csv(loss_path, index=False, encoding='utf-8')
        print(f"训练损失曲线已保存到: {loss_path}")

    # 5. 保存MSE贡献分析汇总
    mse_summary_path = os.path.join(result_dir_path, "mse_contribution_summary.csv")
    mse_summary = pd.DataFrame({
        'Metric': [
            'Total MSE (Train)',
            'Total MSE (Test)',
            'Average Squared Error (Train)',
            'Average Squared Error (Test)',
            'Max MSE Contribution % (Train)',
            'Max MSE Contribution % (Test)',
            'Min MSE Contribution % (Train)',
            'Min MSE Contribution % (Test)',
            'Top 5 MSE Contribution % (Train)',
            'Top 5 MSE Contribution % (Test)'
        ],
        'Value': [
            np.sum(results['train']['squared_errors']),
            np.sum(results['test']['squared_errors']),
            np.mean(results['train']['squared_errors']),
            np.mean(results['test']['squared_errors']),
            results['train']['mse_contributions'].max() * 100,
            results['test']['mse_contributions'].max() * 100,
            results['train']['mse_contributions'].min() * 100,
            results['test']['mse_contributions'].min() * 100,
            np.sum(np.sort(results['train']['mse_contributions'])[-5:]) * 100 if len(
                results['train']['mse_contributions']) >= 5 else np.sum(results['train']['mse_contributions']) * 100,
            np.sum(np.sort(results['test']['mse_contributions'])[-5:]) * 100 if len(
                results['test']['mse_contributions']) >= 5 else np.sum(results['test']['mse_contributions']) * 100
        ]
    })
    mse_summary.to_csv(mse_summary_path, index=False, encoding='utf-8')
    print(f"MSE贡献分析汇总已保存到: {mse_summary_path}")

    return metrics_df


def create_visualizations(results, model, X_train, X_test, feature_columns, target_column):
    """创建可视化图表"""
    print("\n正在创建可视化图表...")

    result_dir_path = result_dir

    # 设置图表样式
    plt.style.use(plot_style)

    # 1. 实际值 vs 预测值散点图
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f'MLP Regression 模型性能可视化\n目标变量: {target_column}\n隐藏层结构: {model.hidden_layer_sizes}',
        fontsize=16, y=1.02)

    datasets = ['train', 'test']
    titles = ['训练集', '测试集']
    colors = ['blue', 'red']

    # 设置是否显示所有点标签
    show_all_labels = len(results['train']['y_true']) + len(results['test']['y_true']) <= max_labels_display

    # 左上：训练集散点图
    ax = axes[0, 0]
    dataset = 'train'
    title = '训练集'
    color = colors[0]

    y_true = results[dataset]['y_true']
    y_pred = results[dataset]['y_pred']
    r2 = results[dataset]['r2']

    # 计算数据范围用于调整标签位置
    data_range = max(y_true.max(), y_pred.max()) - min(y_true.min(), y_pred.min())
    label_offset = data_range * 0.02

    # 散点图
    scatter = ax.scatter(y_true, y_pred, alpha=0.7, color=color, s=80,
                         edgecolors='white', linewidth=0.5)

    # 添加对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', lw=2, label='完美预测线', alpha=0.7)

    # 添加每个点的预测值标签（选择性显示）
    if show_all_labels and len(y_true) <= 30:
        for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
            ax.text(true_val, pred_val + label_offset, f'{pred_val:.3f}',
                    fontsize=7, alpha=0.7, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.2))

    ax.set_xlabel('实际值', fontsize=12)
    ax.set_ylabel('预测值', fontsize=12)
    ax.set_title(f'{title} (R² = {r2:.4f})', fontsize=14, fontweight='bold')

    # 添加统计信息框
    stats_text = f'样本数: {len(y_true)}\nRMSE: {results[dataset]["rmse"]:.4f}\nMAE: {results[dataset]["mae"]:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右上：测试集散点图
    ax = axes[0, 1]
    dataset = 'test'
    title = '测试集'
    color = colors[1]

    y_true = results[dataset]['y_true']
    y_pred = results[dataset]['y_pred']
    r2 = results[dataset]['r2']

    # 计算数据范围用于调整标签位置
    data_range = max(y_true.max(), y_pred.max()) - min(y_true.min(), y_pred.min())
    label_offset = data_range * 0.02

    # 散点图
    scatter = ax.scatter(y_true, y_pred, alpha=0.7, color=color, s=80,
                         edgecolors='white', linewidth=0.5)

    # 添加对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', lw=2, label='完美预测线', alpha=0.7)

    # 添加每个点的预测值标签（选择性显示）
    if show_all_labels and len(y_true) <= 30:
        for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
            ax.text(true_val, pred_val + label_offset, f'{pred_val:.3f}',
                    fontsize=7, alpha=0.7, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.2))

    ax.set_xlabel('实际值', fontsize=12)
    ax.set_ylabel('预测值', fontsize=12)
    ax.set_title(f'{title} (R² = {r2:.4f})', fontsize=14, fontweight='bold')

    # 添加统计信息框
    stats_text = f'样本数: {len(y_true)}\nRMSE: {results[dataset]["rmse"]:.4f}\nMAE: {results[dataset]["mae"]:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 左下：残差分布图
    ax = axes[1, 0]
    all_residuals = []

    for idx, (dataset, title) in enumerate(zip(datasets, titles)):
        color = colors[idx]
        y_true = results[dataset]['y_true']
        y_pred = results[dataset]['y_pred']
        residuals = y_true - y_pred
        all_residuals.extend(residuals)

        # 绘制残差直方图
        ax.hist(residuals, bins=20, alpha=0.5, label=title, color=color, density=True)

    # 添加正态分布曲线
    mu, std = np.mean(all_residuals), np.std(all_residuals)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, label=f'正态分布\n(μ={mu:.2f}, σ={std:.2f})')

    ax.set_xlabel('残差', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title('残差分布图', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右下：性能对比图
    ax = axes[1, 1]
    metrics = ['R²', 'RMSE', 'MAE']
    x = np.arange(len(titles))
    width = 0.25

    r2_values = [results['train']['r2'], results['test']['r2']]
    rmse_values = [results['train']['rmse'], results['test']['rmse']]
    mae_values = [results['train']['mae'], results['test']['mae']]

    bars1 = ax.bar(x - width, r2_values, width, label='R²', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x, rmse_values, width, label='RMSE', color='lightgreen', alpha=0.8)
    bars3 = ax.bar(x + width, mae_values, width, label='MAE', color='salmon', alpha=0.8)

    # 添加数值标签
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    ax.set_xlabel('数据集', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('MLP模型性能指标对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(titles, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加模型信息
    model_info = f"隐藏层: {model.hidden_layer_sizes} | 激活函数: {model.activation} | 学习率: {model.learning_rate}"
    ax.text(0.5, 1.05, model_info, transform=ax.transAxes,
            fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 添加过拟合信息
    if results['overfitting_score'] > 0.1:
        overfit_text = f"注意：模型可能存在过拟合 (训练R²-测试R²={results['overfitting_score']:.3f})"
        ax.text(0.5, -0.15, overfit_text, transform=ax.transAxes,
                fontsize=10, ha='center', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    summary_path = os.path.join(result_dir_path, "model_performance_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"模型性能汇总图已保存到: {summary_path}")

    # 2. 训练损失曲线
    if hasattr(model, 'loss_curve_'):
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(range(1, len(model.loss_curve_) + 1),
                model.loss_curve_, 'b-', linewidth=2)
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=12)
        ax.set_ylabel('损失值 (Loss)', fontsize=12)
        ax.set_title('MLP训练损失曲线', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 标记最佳点
        min_loss_idx = np.argmin(model.loss_curve_)
        ax.scatter(min_loss_idx + 1, model.loss_curve_[min_loss_idx],
                   color='red', s=100, zorder=5,
                   label=f'最小损失: {model.loss_curve_[min_loss_idx]:.4f}')
        ax.legend(fontsize=10)

        loss_path = os.path.join(result_dir_path, "training_loss_curve.png")
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"训练损失曲线已保存到: {loss_path}")

    # 3. 网络结构可视化
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.text(0.5, 0.7, f'网络结构: {results["hidden_layer_sizes"]}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.6, f'总参数数: {results["n_params"]:,}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.5, f'激活函数: {results["activation"]}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.4, f'总层数: {results["n_layers"]}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.3, f'总神经元数: {results["total_neurons"]}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.2, f'过拟合程度: {results["overfitting_score"]:.4f}',
            transform=ax.transAxes, fontsize=14, ha='center')

    ax.set_title('MLP网络结构总结', fontsize=16)
    ax.axis('off')

    network_path = os.path.join(result_dir_path, "network_structure.png")
    plt.savefig(network_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"网络结构图已保存到: {network_path}")

    # 4. 预测误差分布图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, (dataset, title, color) in enumerate(zip(datasets, titles, colors)):
        y_true = results[dataset]['y_true']
        y_pred = results[dataset]['y_pred']
        errors = y_true - y_pred

        axes[idx].hist(errors, bins=30, alpha=0.7, color=color, edgecolor='black')
        axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[idx].set_xlabel('预测误差', fontsize=11)
        axes[idx].set_ylabel('频数', fontsize=11)
        axes[idx].set_title(f'{title}误差分布', fontsize=13, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

        # 添加统计信息
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        axes[idx].text(0.05, 0.95, f'均值: {mean_error:.3f}\n标准差: {std_error:.3f}',
                       transform=axes[idx].transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    error_path = os.path.join(result_dir_path, "error_distribution.png")
    plt.savefig(error_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"误差分布图已保存到: {error_path}")

    # ==================== 5. SHAP值分析图（训练集 + 测试集） ====================
    if save_shap_results:
        try:
            print("\n" + "=" * 60)
            print("开始SHAP值分析")
            print("=" * 60)

            # ====== 准备特征名称 ======
            feature_names = feature_columns  # 直接使用，已经是包含真实名称的列表
            print(f"特征名称 (前5个): {feature_names[:5]}")
            print(f"特征名称数量: {len(feature_names)}")

            # ====== 创建SHAP解释器 ======
            # 使用训练集的一部分作为背景数据
            background_sample = min(100, X_train.shape[0])
            background_data = X_train[:background_sample]

            print(f"背景数据样本数: {background_sample}")
            print("正在创建KernelExplainer...")
            explainer = shap.KernelExplainer(model.predict, background_data)

            # ==================== 计算训练集SHAP值 ====================
            print("\n" + "-" * 40)
            print("计算训练集SHAP值")
            print("-" * 40)

            train_sample_size = min(shap_sample_size, X_train.shape[0])
            X_train_sample = X_train[:train_sample_size]

            print(f"训练集样本数: {train_sample_size}")
            print("计算SHAP值...")
            shap_values_train = explainer.shap_values(X_train_sample)

            # 转换为DataFrame并设置列名
            X_train_df = pd.DataFrame(X_train_sample, columns=feature_names)

            print(f"训练集SHAP值形状: {np.array(shap_values_train).shape}")

            # ==================== 计算测试集SHAP值 ====================
            print("\n" + "-" * 40)
            print("计算测试集SHAP值")
            print("-" * 40)

            test_sample_size = min(shap_sample_size, X_test.shape[0])
            X_test_sample = X_test[:test_sample_size]

            print(f"测试集样本数: {test_sample_size}")
            print("计算SHAP值...")
            shap_values_test = explainer.shap_values(X_test_sample)

            # 转换为DataFrame并设置列名
            X_test_df = pd.DataFrame(X_test_sample, columns=feature_names)

            print(f"测试集SHAP值形状: {np.array(shap_values_test).shape}")

            # ==================== 导出SHAP数据 ====================
            print("\n" + "-" * 40)
            print("导出SHAP数据")
            print("-" * 40)

            try:
                from shap_data_exporter import export_shap_data

                # 导出训练集SHAP数据
                shap_data_dir_train = export_shap_data(
                    explainer=explainer,
                    shap_values=shap_values_train,
                    X_sample=X_train_df,
                    feature_columns=feature_names,
                    result_dir=os.path.join(result_dir_path, "shap_data_train"),
                    max_samples=shap_max_samples,
                    top_n_interactions=shap_top_n_interactions
                )
                print(f"训练集SHAP数据已导出到: {shap_data_dir_train}")

                # 导出测试集SHAP数据
                shap_data_dir_test = export_shap_data(
                    explainer=explainer,
                    shap_values=shap_values_test,
                    X_sample=X_test_df,
                    feature_columns=feature_names,
                    result_dir=os.path.join(result_dir_path, "shap_data_test"),
                    max_samples=shap_max_samples,
                    top_n_interactions=shap_top_n_interactions
                )
                print(f"测试集SHAP数据已导出到: {shap_data_dir_test}")

            except ImportError as e:
                print(f"无法导入SHAP数据导出模块: {e}")
            except Exception as e:
                print(f"导出SHAP数据时出错: {e}")

            # ==================== 保存SHAP值到CSV ====================
            print("\n" + "-" * 40)
            print("保存SHAP值到CSV")
            print("-" * 40)

            try:
                # 训练集SHAP值
                shap_train_df = pd.DataFrame(shap_values_train, columns=feature_names)
                shap_train_df['Sample_Index'] = range(1, len(shap_train_df) + 1)
                shap_train_df['Dataset'] = 'train'
                shap_train_path = os.path.join(result_dir_path, "shap_values_train.csv")
                shap_train_df.to_csv(shap_train_path, index=False, encoding='utf-8')
                print(f"训练集SHAP值已保存到: {shap_train_path}")

                # 测试集SHAP值
                shap_test_df = pd.DataFrame(shap_values_test, columns=feature_names)
                shap_test_df['Sample_Index'] = range(1, len(shap_test_df) + 1)
                shap_test_df['Dataset'] = 'test'
                shap_test_path = os.path.join(result_dir_path, "shap_values_test.csv")
                shap_test_df.to_csv(shap_test_path, index=False, encoding='utf-8')
                print(f"测试集SHAP值已保存到: {shap_test_path}")

            except Exception as e:
                print(f"保存SHAP值失败: {e}")

            # ==================== 生成SHAP可视化图表 ====================
            print("\n" + "-" * 40)
            print("生成SHAP可视化图表")
            print("-" * 40)

            # ---------- 训练集SHAP可视化 ----------
            print("\n生成训练集SHAP可视化...")

            # 训练集 - 特征重要性
            plt.figure(figsize=(14, max(8, len(feature_names) * 0.3)))
            shap.summary_plot(
                shap_values_train,
                X_train_df,
                feature_names=feature_names,
                plot_type="bar",
                max_display=len(feature_names),
                show=False
            )
            plt.title(f'SHAP特征重要性 (训练集) - MLP', fontsize=14, fontweight='bold')
            plt.tight_layout()
            shap_importance_train_path = os.path.join(result_dir_path, "shap_feature_importance_train.png")
            plt.savefig(shap_importance_train_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"训练集SHAP特征重要性图已保存到: {shap_importance_train_path}")

            # 训练集 - summary plot
            plt.figure(figsize=(14, max(8, len(feature_names) * 0.3)))
            shap.summary_plot(
                shap_values_train,
                X_train_df,
                feature_names=feature_names,
                max_display=len(feature_names),
                show=False
            )
            plt.title(f'SHAP summary plot (训练集) - MLP', fontsize=14, fontweight='bold')
            plt.tight_layout()
            shap_summary_train_path = os.path.join(result_dir_path, "shap_summary_plot_train.png")
            plt.savefig(shap_summary_train_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"训练集SHAP影响图已保存到: {shap_summary_train_path}")

            # ---------- 测试集SHAP可视化 ----------
            print("\n生成测试集SHAP可视化...")

            # 测试集 - 特征重要性
            plt.figure(figsize=(14, max(8, len(feature_names) * 0.3)))
            shap.summary_plot(
                shap_values_test,
                X_test_df,
                feature_names=feature_names,
                plot_type="bar",
                max_display=len(feature_names),
                show=False
            )
            plt.title(f'SHAP特征重要性 (测试集) - MLP', fontsize=14, fontweight='bold')
            plt.tight_layout()
            shap_importance_test_path = os.path.join(result_dir_path, "shap_feature_importance_test.png")
            plt.savefig(shap_importance_test_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"测试集SHAP特征重要性图已保存到: {shap_importance_test_path}")

            # 测试集 - summary plot
            plt.figure(figsize=(14, max(8, len(feature_names) * 0.3)))
            shap.summary_plot(
                shap_values_test,
                X_test_df,
                feature_names=feature_names,
                max_display=len(feature_names),
                show=False
            )
            plt.title(f'SHAP summary plot (测试集) - MLP', fontsize=14, fontweight='bold')
            plt.tight_layout()
            shap_summary_test_path = os.path.join(result_dir_path, "shap_summary_plot_test.png")
            plt.savefig(shap_summary_test_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"测试集SHAP影响图已保存到: {shap_summary_test_path}")

            print("\n" + "=" * 60)
            print("SHAP分析完成!")
            print("=" * 60)

        except Exception as e:
            print(f"SHAP分析时出现错误: {e}")
            import traceback
            traceback.print_exc()
            print("跳过SHAP分析，请确保已安装shap库: pip install shap")


def main():
    """主函数"""
    # 打印配置信息
    print_config()

    # 确保所有目录存在
    ensure_directories()

    print("开始 MLP Regression 建模流程")
    print("=" * 60)

    # 1. 加载和准备数据
    (X_train, y_train, train_names,
     X_test, y_test, test_names,
     feature_columns, target_column) = load_and_prepare_data()

    # 2. 数据标准化（对MLP非常重要）
    print("\n正在标准化数据...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. 网格搜索和交叉验证
    grid_search = perform_grid_search(X_train_scaled, X_train, y_train, train_names, feature_columns, target_column)
    best_model = grid_search.best_estimator_

    # 4. 评估模型
    results = evaluate_model(best_model,
                             X_train_scaled, y_train,
                             X_test_scaled, y_test)

    # 5. 保存结果
    metrics_df = save_results(best_model, results, feature_columns, target_column, train_names, test_names)

    # 6. 创建可视化
    if create_visualizations:
        create_visualizations(results, best_model, X_train_scaled, X_test_scaled, feature_columns, target_column)

    # 7. 打印总结
    print("\n" + "=" * 60)
    print("MLP建模流程完成!")
    print("=" * 60)
    print(f"最佳模型参数:")
    print(f"  隐藏层结构: {best_model.hidden_layer_sizes}")
    print(f"  激活函数: {best_model.activation}")
    print(f"  正则化参数 (alpha): {best_model.alpha}")
    print(f"  学习率: {best_model.learning_rate}")
    print(f"  初始学习率: {best_model.learning_rate_init}")
    print(f"  最大迭代次数: {best_model.max_iter}")
    print(f"  批次大小: {best_model.batch_size}")
    print(f"\n模型在测试集上的表现:")
    print(f"  R²: {results['test']['r2']:.4f}")
    print(f"  MSE: {results['test']['mse']:.4f}")
    print(f"  RMSE: {results['test']['rmse']:.4f}")
    print(f"  MAE: {results['test']['mae']:.4f}")
    print(f"\n网络结构:")
    print(f"  总层数: {results['n_layers']}")
    print(f"  总神经元数: {results['total_neurons']}")
    print(f"  近似参数数量: {results['n_params']:,}")
    print(f"\n模型诊断信息:")
    print(f"  过拟合程度: {results['overfitting_score']:.4f}")
    print(f"\nMSE贡献分析:")
    print(f"  训练集总MSE: {np.sum(results['train']['squared_errors']):.4f}")
    print(f"  测试集总MSE: {np.sum(results['test']['squared_errors']):.4f}")
    print(f"  训练集最大MSE贡献: {results['train']['mse_contributions'].max() * 100:.2f}%")
    print(f"  测试集最大MSE贡献: {results['test']['mse_contributions'].max() * 100:.2f}%")

    print(f"\n所有结果已保存到: {result_dir}")
    print(f"  - 交叉验证数据: {cv_dir}")
    print(f"  - SHAP数据: {shap_dir}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        import shap

        print("shap库已安装")
    except ImportError:
        print("警告：shap库未安装，SHAP分析将被跳过")
        print("安装命令: pip install shap")

    main()