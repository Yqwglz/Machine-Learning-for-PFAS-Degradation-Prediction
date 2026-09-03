import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# ===================== 配置区域（所有可调参数集中管理）=====================
# ============================================================================

# -------------------- 基础路径配置 --------------------
result_dir = "./result_Major revision/LOCO/ridge/905"
cv_dir = os.path.join(result_dir, "cv5")
shap_dir = os.path.join(result_dir, "shap_data")

# -------------------- 数据路径配置 --------------------
train_path = './Data_processed/LOGO/LOCO/train905.csv'
test_path = './Data_processed/LOGO/LOCO/test905.csv'
# 目标列名
target_column = "Kobs(h-1)"

# 要排除的列（不作为特征）
exclude_columns = ["Name", target_column]

# -------------------- 随机种子配置 --------------------
random_seed = 42

# -------------------- 交叉验证配置 --------------------
cv_n_splits = 5
cv_shuffle = True

# -------------------- Ridge网格搜索参数 --------------------
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

# -------------------- Ridge模型固定参数 --------------------
model_params = {
    'random_state': 42
}

# -------------------- GridSearchCV配置 --------------------
grid_search_params = {
    'scoring': 'neg_mean_squared_error',
    'n_jobs': -1,
    'verbose': 1,
    'refit': True
}

# -------------------- 可视化配置 --------------------
font_sans_serif = ['SimSun', 'Times New Roman']
plot_style = 'seaborn-v0_8-darkgrid'
max_labels_display = 50
max_features_display = 50

# -------------------- 输出开关 --------------------
save_cv_data = True
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
    print("Ridge 模型配置信息")
    print("=" * 70)
    print(f"基础结果目录: {result_dir}")
    print(f"训练数据路径: {train_path}")
    print(f"测试数据路径: {test_path}")
    print(f"目标列: {target_column}")
    print(f"随机种子: {random_seed}")
    print(f"交叉验证折数: {cv_n_splits}")
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

    print("\n训练集列名:")
    print(train_df.columns.tolist())

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


def perform_grid_search(X_train, y_train, train_names=None, feature_columns=None, target_column=None):
    """执行网格搜索进行超参数调优，并导出交叉验证各折数据"""
    print("\n正在执行网格搜索...")

    # 使用配置中的KFold参数
    kfold = KFold(n_splits=cv_n_splits,
                  shuffle=cv_shuffle,
                  random_state=random_seed)

    # ========== 导出各折数据 ==========
    if save_cv_data:
        print("\n正在导出交叉验证各折数据...")

        cv_dir_path = cv_dir
        os.makedirs(cv_dir_path, exist_ok=True)

        # 准备数据
        X_df = pd.DataFrame(X_train, columns=feature_columns)
        y_series = pd.Series(y_train, name=target_column)

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

        all_folds = {}

        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(X_train), 1):
            print(f"  第 {fold_idx} 折: 训练集 {len(train_indices)} 样本, 验证集 {len(val_indices)} 样本")

            train_df = X_df.iloc[train_indices].copy()
            train_df[target_column] = y_series.iloc[train_indices].values
            if name_series is not None:
                train_df['Name'] = name_series.iloc[train_indices].values
            train_df['Fold'] = fold_idx
            train_df['Dataset_Type'] = 'Training'

            val_df = X_df.iloc[val_indices].copy()
            val_df[target_column] = y_series.iloc[val_indices].values
            if name_series is not None:
                val_df['Name'] = name_series.iloc[val_indices].values
            val_df['Fold'] = fold_idx
            val_df['Dataset_Type'] = 'Validation'

            fold_df = pd.concat([train_df, val_df], ignore_index=True)

            if name_series is not None:
                cols = ['Name'] + [col for col in fold_df.columns if col != 'Name']
                fold_df = fold_df[cols]

            all_folds[f'Fold_{fold_idx}'] = fold_df

        # 保存为CSV文件
        os.makedirs(cv_dir_path, exist_ok=True)
        for sheet_name, df in all_folds.items():
            df.to_csv(os.path.join(cv_dir_path, f"{sheet_name}.csv"), index=False)
        print(f"  ✓ CSV文件已保存到: {cv_dir_path}\n")

    # ========== 执行网格搜索 ==========
    ridge = Ridge(**model_params)

    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=kfold,
        **grid_search_params
    )

    grid_search.fit(X_train, y_train)

    print("\n网格搜索完成!")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数 (负MSE): {grid_search.best_score_:.4f}")

    return grid_search


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """评估模型性能"""
    print("\n正在评估模型性能...")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)

    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 计算MSE贡献
    train_squared_errors = (y_train - y_train_pred) ** 2
    test_squared_errors = (y_test - y_test_pred) ** 2

    train_total_mse = np.sum(train_squared_errors)
    test_total_mse = np.sum(test_squared_errors)

    train_mse_contributions = train_squared_errors / train_total_mse if train_total_mse > 0 else np.zeros_like(
        train_squared_errors)
    test_mse_contributions = test_squared_errors / test_total_mse if test_total_mse > 0 else np.zeros_like(
        test_squared_errors)

    overfitting_score = train_r2 - test_r2

    # 计算系数统计
    coef_abs = np.abs(model.coef_)
    mean_coef_abs = np.mean(coef_abs)
    std_coef_abs = np.std(coef_abs)

    print("\n" + "=" * 70)
    print("Ridge Regression 模型性能评估结果")
    print("=" * 70)
    print(f"{'数据集':<10} {'R²':<12} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'过拟合度':<12}")
    print(f"{'-' * 70}")
    print(f"{'训练集':<10} {train_r2:<12.4f} {train_mse:<12.4f} {train_rmse:<12.4f} {train_mae:<12.4f} {'-' * 12}")
    print(f"{'测试集':<10} {test_r2:<12.4f} {test_mse:<12.4f} {test_rmse:<12.4f} {test_mae:<12.4f} {overfitting_score:<12.4f}")
    print("=" * 70)
    print(f"\n模型系数统计:")
    print(f"  平均系数绝对值: {mean_coef_abs:.6f}")
    print(f"  系数标准差: {std_coef_abs:.6f}")
    print(f"  最大系数绝对值: {np.max(coef_abs):.6f}")
    print(f"  最小系数绝对值: {np.min(coef_abs):.6f}")

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
        'overfitting_score': overfitting_score,
        'mean_coef_abs': mean_coef_abs,
        'std_coef_abs': std_coef_abs,
        'max_coef_abs': np.max(coef_abs),
        'min_coef_abs': np.min(coef_abs)
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
        f.write("Ridge Regression 模型参数\n")
        f.write("=" * 70 + "\n\n")

        f.write("超参数:\n")
        f.write(f"  alpha (正则化强度): {model.alpha}\n")
        f.write(f"  solver: {model.solver}\n")
        f.write(f"  fit_intercept: {model.fit_intercept}\n")
        f.write(f"  copy_X: {model.copy_X}\n")
        f.write(f"  max_iter: {model.max_iter}\n")
        f.write(f"  tol: {model.tol}\n\n")

        f.write("模型性能:\n")
        f.write(f"  训练集 R²: {results['train']['r2']:.4f}\n")
        f.write(f"  测试集 R²: {results['test']['r2']:.4f}\n")
        f.write(f"  过拟合程度: {results['overfitting_score']:.4f}\n")
        if results['overfitting_score'] > 0.1:
            f.write(f"  ⚠️ 警告：模型可能存在过拟合\n")
        f.write(f"  平均系数绝对值: {results['mean_coef_abs']:.6f}\n")
        f.write(f"  系数标准差: {results['std_coef_abs']:.6f}\n\n")

        f.write(f"\nMSE贡献分析:\n")
        f.write(f"  训练集总MSE: {results['train']['mse'] * len(results['train']['y_true']):.4f}\n")
        f.write(f"  测试集总MSE: {results['test']['mse'] * len(results['test']['y_true']):.4f}\n")
        f.write(
            f"  训练集样本MSE贡献范围: {results['train']['mse_contributions'].min():.6f} - {results['train']['mse_contributions'].max():.6f}\n")
        f.write(
            f"  测试集样本MSE贡献范围: {results['test']['mse_contributions'].min():.6f} - {results['test']['mse_contributions'].max():.6f}\n\n")

        f.write("模型系数:\n")
        if hasattr(model, 'coef_'):
            f.write(f"  截距 (intercept): {model.intercept_:.6f}\n\n")
            f.write("特征系数 (前20个):\n")

            # 按系数绝对值排序
            coef_df = pd.DataFrame({
                'Feature': feature_columns,
                'Coefficient': model.coef_,
                'Abs_Coefficient': np.abs(model.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)

            for i, row in coef_df.head(20).iterrows():
                f.write(f"  {i + 1:3d}. {row['Feature']:<30}: {row['Coefficient']:.6f} (绝对值: {row['Abs_Coefficient']:.6f})\n")

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
        'Alpha': [model.alpha, model.alpha],
        'Solver': [model.solver, model.solver]
    })
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8')
    print(f"性能指标已保存到: {metrics_path}")

    # 3. 保存预测结果（包含MSE贡献分析）
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

    predictions_df = predictions_df.sort_values('MSE_Contribution_Percent', ascending=False)
    predictions_df['Rank_by_MSE_Contribution'] = range(1, len(predictions_df) + 1)

    pred_path = os.path.join(result_dir_path, "predictions.csv")
    predictions_df.to_csv(pred_path, index=False, encoding='utf-8')
    print(f"预测结果（含MSE贡献分析）已保存到: {pred_path}")

    # 4. 保存系数结果
    coef_df = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_),
        'Rank': np.argsort(np.argsort(-np.abs(model.coef_))) + 1
    }).sort_values('Abs_Coefficient', ascending=False)

    coef_path = os.path.join(result_dir_path, "feature_coefficients.csv")
    coef_df.to_csv(coef_path, index=False, encoding='utf-8')
    print(f"特征系数已保存到: {coef_path}")

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
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f'Ridge Regression 模型性能可视化\n目标变量: {target_column}\nAlpha={model.alpha}, Solver={model.solver}',
        fontsize=16, y=1.02)

    datasets = ['train', 'test']
    titles = ['训练集', '测试集']
    colors = ['blue', 'red']

    show_all_labels = len(results['train']['y_true']) + len(results['test']['y_true']) <= max_labels_display

    for idx, (dataset, title, color) in enumerate(zip(datasets, titles, colors)):
        ax = axes[idx]
        y_true = results[dataset]['y_true']
        y_pred = results[dataset]['y_pred']
        r2 = results[dataset]['r2']

        data_range = max(y_true.max(), y_pred.max()) - min(y_true.min(), y_pred.min())
        label_offset = data_range * 0.02

        scatter = ax.scatter(y_true, y_pred, alpha=0.7, color=color, s=80,
                             edgecolors='white', linewidth=0.5)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', lw=2, label='完美预测线', alpha=0.7)

        if show_all_labels:
            for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
                ax.text(true_val, pred_val + label_offset, f'{pred_val:.3f}',
                        fontsize=7, alpha=0.7, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.2))
        else:
            residuals = np.abs(y_true - y_pred)
            top_n = min(5, len(residuals))
            if top_n > 0:
                top_indices = np.argsort(residuals)[-top_n:]
                for i in top_indices:
                    ax.text(y_true[i], y_pred[i] + label_offset, f'{y_pred[i]:.3f}',
                            fontsize=8, alpha=0.8, ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.3))

        ax.set_xlabel('实际值', fontsize=12)
        ax.set_ylabel('预测值', fontsize=12)
        ax.set_title(f'{title} (R² = {r2:.4f})', fontsize=14, fontweight='bold')

        stats_text = f'样本数: {len(y_true)}\nRMSE: {results[dataset]["rmse"]:.4f}\nMAE: {results[dataset]["mae"]:.4f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_path = os.path.join(result_dir_path, "predictions_scatter_with_labels.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"带预测值标签的散点图已保存到: {scatter_path}")

    # 2. 残差分布图
    fig, ax = plt.subplots(figsize=(10, 6))

    all_residuals = []
    for dataset, title, color in zip(datasets, titles, colors):
        y_true = results[dataset]['y_true']
        y_pred = results[dataset]['y_pred']
        residuals = y_true - y_pred
        all_residuals.extend(residuals)

        ax.hist(residuals, bins=20, alpha=0.5, label=title, color=color, density=True)

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

    plt.tight_layout()
    residual_path = os.path.join(result_dir_path, "residual_distribution.png")
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"残差分布图已保存到: {residual_path}")

    # 3. 特征重要性图（系数绝对值）
    if hasattr(model, 'coef_'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

        # 按系数绝对值排序
        coef_df = pd.DataFrame({
            'Feature': feature_columns,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=True)

        top_n = min(20, len(coef_df))
        top_coef = coef_df.tail(top_n)

        # 图1：系数条形图
        colors_coef = ['red' if x < 0 else 'blue' for x in top_coef['Coefficient']]
        y_pos = np.arange(top_n)

        bars1 = ax1.barh(y_pos, top_coef['Coefficient'], color=colors_coef, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_coef['Feature'], fontsize=9)
        ax1.set_xlabel('系数值', fontsize=12)
        ax1.set_title(f'Top {top_n} 特征系数 (Ridge)', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='black', linewidth=0.5)

        for i, (bar, coef) in enumerate(zip(bars1, top_coef['Coefficient'])):
            ax1.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                     f'{coef:.6f}', ha='left' if coef >= 0 else 'right', va='center', fontsize=7)

        # 图2：系数绝对值条形图
        bars2 = ax2.barh(y_pos, top_coef['Abs_Coefficient'], color='lightcoral', alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_coef['Feature'], fontsize=9)
        ax2.set_xlabel('系数绝对值', fontsize=12)
        ax2.set_title(f'Top {top_n} 特征重要性（系数绝对值）', fontsize=14, fontweight='bold')

        for i, (bar, abs_coef) in enumerate(zip(bars2, top_coef['Abs_Coefficient'])):
            ax2.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                     f'{abs_coef:.6f}', ha='left', va='center', fontsize=7)

        plt.tight_layout()
        coef_path = os.path.join(result_dir_path, "feature_coefficients.png")
        plt.savefig(coef_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特征系数图已保存到: {coef_path}")

    # 4. 性能对比图
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['R²', 'RMSE', 'MAE']
    datasets_plot = ['训练集', '测试集']

    r2_values = [results['train']['r2'], results['test']['r2']]
    rmse_values = [results['train']['rmse'], results['test']['rmse']]
    mae_values = [results['train']['mae'], results['test']['mae']]

    x = np.arange(len(datasets_plot))
    width = 0.25

    bars1 = ax.bar(x - width, r2_values, width, label='R²', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x, rmse_values, width, label='RMSE', color='lightgreen', alpha=0.8)
    bars3 = ax.bar(x + width, mae_values, width, label='MAE', color='salmon', alpha=0.8)

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
    ax.set_title('Ridge模型性能指标对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_plot, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    model_info = f"Alpha: {model.alpha} | Solver: {model.solver} | 过拟合度: {results['overfitting_score']:.4f}"
    ax.text(0.5, 1.05, model_info, transform=ax.transAxes,
            fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    if results['overfitting_score'] > 0.1:
        overfit_text = f"注意：模型可能存在过拟合 (训练R²-测试R²={results['overfitting_score']:.3f})"
        ax.text(0.5, -0.15, overfit_text, transform=ax.transAxes,
                fontsize=10, ha='center', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    metrics_path = os.path.join(result_dir_path, "performance_metrics.png")
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"性能指标图已保存到: {metrics_path}")

    # 5. 模型参数总结图
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.text(0.5, 0.7, f'正则化强度 (Alpha): {model.alpha}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.6, f'求解器 (Solver): {model.solver}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.5, f'训练集 R²: {results["train"]["r2"]:.4f}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.4, f'测试集 R²: {results["test"]["r2"]:.4f}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.3, f'过拟合程度: {results["overfitting_score"]:.4f}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.2, f'平均系数绝对值: {results["mean_coef_abs"]:.6f}',
            transform=ax.transAxes, fontsize=14, ha='center')

    ax.set_title('Ridge模型参数总结', fontsize=16)
    ax.axis('off')

    params_summary_path = os.path.join(result_dir_path, "model_parameters_summary.png")
    plt.savefig(params_summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"模型参数总结图已保存到: {params_summary_path}")


def main():
    """主函数"""
    # 打印配置信息
    print_config()

    # 确保所有目录存在
    ensure_directories()

    print("开始 Ridge Regression 建模流程")
    print("=" * 70)
    print("Ridge回归是一种L2正则化的线性回归模型")
    print("特点：通过惩罚系数大小来防止过拟合")
    print("应用场景：多重共线性、高维数据、特征选择")
    print("=" * 70)

    # 1. 加载和准备数据
    (X_train, y_train, train_names,
     X_test, y_test, test_names,
     feature_columns, target_column) = load_and_prepare_data()

    # 2. 数据标准化
    print("\n正在标准化数据...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. 网格搜索和交叉验证
    grid_search = perform_grid_search(X_train_scaled, y_train, train_names, feature_columns, target_column)
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
    print("\n" + "=" * 70)
    print("Ridge Regression 建模流程完成!")
    print("=" * 70)
    print(f"最佳模型参数:")
    print(f"  alpha: {best_model.alpha}")
    print(f"  solver: {best_model.solver}")
    print(f"\n模型在测试集上的表现:")
    print(f"  R²: {results['test']['r2']:.4f}")
    print(f"  MSE: {results['test']['mse']:.4f}")
    print(f"  RMSE: {results['test']['rmse']:.4f}")
    print(f"  MAE: {results['test']['mae']:.4f}")
    print(f"\n模型诊断信息:")
    print(f"  过拟合程度: {results['overfitting_score']:.4f}")
    print(f"  平均系数绝对值: {results['mean_coef_abs']:.6f}")
    print(f"\nMSE贡献分析:")
    print(f"  训练集总MSE: {np.sum(results['train']['squared_errors']):.4f}")
    print(f"  测试集总MSE: {np.sum(results['test']['squared_errors']):.4f}")
    print(f"  训练集最大MSE贡献: {results['train']['mse_contributions'].max() * 100:.2f}%")
    print(f"  测试集最大MSE贡献: {results['test']['mse_contributions'].max() * 100:.2f}%")

    if results['overfitting_score'] > 0.1:
        print(f"  ⚠️ 警告：模型可能存在过拟合，建议:")
        print(f"    1. 增加alpha值加强正则化")
        print(f"    2. 尝试不同的solver")

    print(f"\n所有结果已保存到: {result_dir}")
    print(f"  - 交叉验证数据: {cv_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()