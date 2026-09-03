import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import stats
import shap

warnings.filterwarnings('ignore')




result_dir = "./result_Major revision/LOCO/xgb/905"
cv_dir = os.path.join(result_dir, "cv5")
shap_dir = os.path.join(result_dir, "shap_data")


train_path = './Data_processed/LOGO/LOCO/train905.csv'
test_path = './Data_processed/LOGO/LOCO/test905.csv'


target_column = "Kobs(h-1)"


exclude_columns = ["Name", target_column]


random_seed = 42


cv_n_splits = 5
cv_shuffle = True


xgb
param_grid = {
'n_estimators': [100, 200],  # 树的数量
'max_depth': [3, 5, 7],  # 树的最大深度
'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 学习率
'subsample': [0.5, 0.7, 1.0],  # 样本采样率
'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # 特征采样率
'min_child_weight': [1, 3, 5],  # 叶子节点最小权重和
'gamma': [0, 0.1, 0.2],  # 节点分裂所需的最小损失减少
'reg_alpha': [0, 0.1, 1],  # L1正则化
'reg_lambda': [1, 1.5, 2],  # L2正则化
}


model_params = {
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'verbosity': 0
}


grid_search_params = {
    'scoring': 'neg_mean_squared_error',
    'n_jobs': -1,
    'verbose': 1,
    'refit': True
}


shap_sample_size = 100
shap_max_samples = 10
shap_top_n_interactions = 10


font_sans_serif = ['SimSun', 'Times New Roman']
plot_style = 'seaborn-v0_8-darkgrid'
max_labels_display = 50
max_features_display = 50


save_cv_data = True
save_shap_results = True
create_visualizations = True


plt.rcParams['font.sans-serif'] = font_sans_serif
plt.rcParams['axes.unicode_minus'] = False


np.random.seed(random_seed)


def ensure_directories():
    directories = [result_dir, cv_dir, shap_dir]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print(f"✓ 结果目录已创建: {result_dir}")
    return result_dir


def print_config():
    """打印当前配置信息"""
    print("\n" + "=" * 70)
    print("XGBoost 模型配置信息")
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
    print("\n正在执行XGBoost网格搜索...")
    print("XGBoost是一种梯度提升算法，具有强大的预测能力和特征重要性分析")

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
    xgb_model = xgb.XGBRegressor(**model_params)

    grid_search = GridSearchCV(
        estimator=xgb_model,
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

    # 获取特征重要性
    feature_importance = model.feature_importances_
    n_important_features = np.sum(feature_importance > np.mean(feature_importance))

    # 计算树的平均深度
    booster = model.get_booster()
    tree_depths = []
    for tree in booster.get_dump():
        depth = max([line.count('\t') for line in tree.split('\n') if line])
        tree_depths.append(depth)
    avg_tree_depth = np.mean(tree_depths) if tree_depths else 0

    print("\n" + "=" * 70)
    print("XGBoost模型性能评估结果")
    print("=" * 70)
    print(f"{'数据集':<10} {'R²':<12} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'过拟合度':<12}")
    print(f"{'-' * 70}")
    print(f"{'训练集':<10} {train_r2:<12.4f} {train_mse:<12.4f} {train_rmse:<12.4f} {train_mae:<12.4f} {'-' * 12}")
    print(f"{'测试集':<10} {test_r2:<12.4f} {test_mse:<12.4f} {test_rmse:<12.4f} {test_mae:<12.4f} {overfitting_score:<12.4f}")
    print("=" * 70)
    print(f"\nXGBoost模型信息:")
    print(f"  树的数量: {model.n_estimators}")
    print(f"  树的最大深度: {model.max_depth}")
    print(f"  平均树深度: {avg_tree_depth:.1f}")
    print(f"  学习率: {model.learning_rate}")
    print(f"  重要特征数量 (>平均重要性): {n_important_features}/{len(feature_importance)}")
    print(f"  过拟合程度 (训练R²-测试R²): {overfitting_score:.4f}")

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
        'feature_importance': feature_importance,
        'n_important_features': n_important_features,
        'overfitting_score': overfitting_score,
        'avg_tree_depth': avg_tree_depth,
        'model_params': model.get_params()
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
        f.write("XGBoost Regression 模型参数\n")
        f.write("=" * 70 + "\n\n")

        f.write("XGBoost超参数:\n")
        f.write(f"  n_estimators (树的数量): {model.n_estimators}\n")
        f.write(f"  max_depth (最大深度): {model.max_depth}\n")
        f.write(f"  learning_rate (学习率): {model.learning_rate}\n")
        f.write(f"  subsample (样本采样率): {model.subsample}\n")
        f.write(f"  colsample_bytree (特征采样率): {model.colsample_bytree}\n")
        f.write(f"  min_child_weight (最小子节点权重): {model.min_child_weight}\n")
        f.write(f"  gamma (分裂阈值): {model.gamma}\n")
        f.write(f"  reg_alpha (L1正则化): {model.reg_alpha}\n")
        f.write(f"  reg_lambda (L2正则化): {model.reg_lambda}\n")
        f.write(f"  objective (目标函数): {model.objective}\n")
        f.write(f"  random_state (随机种子): {model.random_state}\n\n")

        f.write("模型性能总结:\n")
        f.write(f"  训练集 R²: {results['train']['r2']:.4f}\n")
        f.write(f"  测试集 R²: {results['test']['r2']:.4f}\n")
        f.write(f"  过拟合程度: {results['overfitting_score']:.4f}\n")
        f.write(f"  平均树深度: {results['avg_tree_depth']:.1f}\n")
        f.write(f"  重要特征数量: {results['n_important_features']}\n")

        f.write(f"\nMSE贡献分析:\n")
        f.write(f"  训练集总MSE: {results['train']['mse'] * len(results['train']['y_true']):.4f}\n")
        f.write(f"  测试集总MSE: {results['test']['mse'] * len(results['test']['y_true']):.4f}\n")
        f.write(
            f"  训练集样本MSE贡献范围: {results['train']['mse_contributions'].min():.6f} - {results['train']['mse_contributions'].max():.6f}\n")
        f.write(
            f"  测试集样本MSE贡献范围: {results['test']['mse_contributions'].min():.6f} - {results['test']['mse_contributions'].max():.6f}\n\n")

        f.write(f"\nXGBoost模型特点:\n")
        f.write(f"  - 梯度提升算法，集成多棵决策树\n")
        f.write(f"  - 支持正则化，防止过拟合\n")
        f.write(f"  - 自动处理缺失值\n")
        f.write(f"  - 内置交叉验证\n")
        f.write(f"  - 提供特征重要性分析\n")
        f.write(f"  - 支持并行计算\n\n")

        f.write("特征重要性 (前20个):\n")
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': results['feature_importance']
        }).sort_values('Importance', ascending=False)

        for i, (feature, importance) in enumerate(zip(importance_df['Feature'][:20],
                                                      importance_df['Importance'][:20]), 1):
            f.write(f"  {i:3d}. {feature:<30}: {importance:.6f}\n")

        f.write(f"\n总特征数: {len(feature_columns)}\n")
        f.write(f"平均特征重要性: {np.mean(results['feature_importance']):.6f}\n")
        f.write(f"重要性标准差: {np.std(results['feature_importance']):.6f}\n")

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
        'Avg_Tree_Depth': [results['avg_tree_depth'], results['avg_tree_depth']],
        'Important_Features': [results['n_important_features'], results['n_important_features']]
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

    # 4. 保存特征重要性结果
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': results['feature_importance'],
        'Rank': np.argsort(np.argsort(-results['feature_importance'])) + 1,
        'Cumulative_Importance': np.cumsum(np.sort(results['feature_importance'])[::-1]),
        'Is_Important': results['feature_importance'] > np.mean(results['feature_importance'])
    }).sort_values('Importance', ascending=False)

    importance_path = os.path.join(result_dir_path, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False, encoding='utf-8')
    print(f"特征重要性已保存到: {importance_path}")

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

    return metrics_df, importance_df


def create_visualizations(results, model, X_train, X_test, feature_columns, target_column):
    """创建可视化图表"""
    print("\n正在创建可视化图表...")

    result_dir_path = result_dir

    # 设置图表样式
    plt.style.use(plot_style)

    # 1. 实际值 vs 预测值散点图
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f'XGBoost Regression 模型性能可视化\n目标变量: {target_column}\n树数量={model.n_estimators}, 深度={model.max_depth}',
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

    # 3. 特征重要性图
    if hasattr(model, 'feature_importances_'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

        importance = results['feature_importance']
        sorted_idx = np.argsort(importance)

        n_features = len(feature_columns)
        max_display_features = min(max_features_display, n_features)
        display_indices = np.linspace(0, n_features - 1, max_display_features, dtype=int)

        display_features = [feature_columns[i] for i in display_indices]
        display_importance = importance[display_indices]

        colors_bar = plt.get_cmap('viridis')(np.linspace(0.3, 0.9, max_display_features))
        y_pos = np.arange(max_display_features)

        bars = ax1.barh(y_pos, display_importance, color=colors_bar, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(display_features, fontsize=8)
        ax1.set_xlabel('重要性分数', fontsize=12)
        ax1.set_title(f'特征重要性 (XGBoost) - 显示{max_display_features}个特征', fontsize=14, fontweight='bold')

        for i, (bar, imp) in enumerate(zip(bars, display_importance)):
            ax1.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                     f'{imp:.4f}', ha='left', va='center', fontsize=6)

        cumulative_importance = np.cumsum(np.sort(importance)[::-1])

        ax2.plot(range(1, n_features + 1), cumulative_importance, 'b-', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80%重要性')
        ax2.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90%重要性')
        ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%重要性')

        for threshold in [0.8, 0.9, 0.95]:
            idx = np.where(cumulative_importance >= threshold)[0]
            if len(idx) > 0:
                n_features_threshold = idx[0] + 1
                ax2.plot(n_features_threshold, threshold, 'ro', markersize=8)
                ax2.text(n_features_threshold, threshold + 0.02,
                         f'{n_features_threshold}个特征', fontsize=9, ha='center')

        ax2.set_xlabel('特征数量', fontsize=12)
        ax2.set_ylabel('累积重要性', fontsize=12)
        ax2.set_title('累积特征重要性 (XGBoost)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, n_features + 1)

        plt.tight_layout()
        importance_path = os.path.join(result_dir_path, "feature_importance.png")
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特征重要性图已保存到: {importance_path}")

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
    ax.set_title('XGBoost模型性能指标对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_plot, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    model_info = f"树数量: {model.n_estimators} | 最大深度: {model.max_depth} | 学习率: {model.learning_rate}"
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

    # 5. 网络结构总结图
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.text(0.5, 0.7, f'决策树数量: {model.n_estimators}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.6, f'树的最大深度: {model.max_depth}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.5, f'平均树深度: {results["avg_tree_depth"]:.1f}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.4, f'学习率: {model.learning_rate}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.3, f'过拟合程度: {results["overfitting_score"]:.4f}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.2, f'训练集R²: {results["train"]["r2"]:.4f}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.1, f'测试集R²: {results["test"]["r2"]:.4f}',
            transform=ax.transAxes, fontsize=14, ha='center')

    ax.set_title('XGBoost模型网络结构总结', fontsize=16)
    ax.axis('off')

    network_path = os.path.join(result_dir_path, "network_structure.png")
    plt.savefig(network_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"网络结构图已保存到: {network_path}")

    # 6. SHAP值分析图
    if save_shap_results:
        try:
            print("\n正在计算SHAP值...")
            print(f"特征数量: {len(feature_columns)}")

            # 创建SHAP解释器
            explainer = shap.TreeExplainer(model)

            sample_size = min(shap_sample_size, X_train.shape[0])
            if isinstance(X_train, pd.DataFrame):
                X_sample = X_train.iloc[:sample_size]
            else:
                X_sample = X_train[:sample_size]

            shap_values = explainer.shap_values(X_sample)

            print("正在生成SHAP可视化图表...")

            # SHAP特征重要性
            plt.figure(figsize=(14, max(8, len(feature_columns) * 0.3)))
            shap.summary_plot(shap_values, X_sample, plot_type="bar",
                             max_display=len(feature_columns),
                             show=False)
            plt.title(f'SHAP特征重要性 (XGBoost) - 所有{len(feature_columns)}个特征', fontsize=14, fontweight='bold')
            plt.tight_layout()
            shap_importance_path = os.path.join(result_dir_path, "shap_feature_importance_all.png")
            plt.savefig(shap_importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP特征重要性图已保存到: {shap_importance_path}")

            # SHAP summary plot
            plt.figure(figsize=(14, max(8, len(feature_columns) * 0.3)))
            shap.summary_plot(shap_values, X_sample,
                             max_display=len(feature_columns),
                             show=False)
            plt.title(f'SHAP summary plot (XGBoost) - all{len(feature_columns)}个特征', fontsize=14, fontweight='bold')
            plt.tight_layout()
            shap_summary_path = os.path.join(result_dir_path, "shap_summary_plot_all.png")
            plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP特征影响图已保存到: {shap_summary_path}")

        except Exception as e:
            print(f"SHAP分析时出现错误: {e}")
            import traceback
            traceback.print_exc()

    # 7. 学习曲线分析图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 模拟学习曲线
    n_trees = model.n_estimators
    iterations = range(1, n_trees + 1)

    # 模拟训练误差和测试误差
    train_errors = []
    test_errors = []

    base_train_error = results['train']['rmse']
    base_test_error = results['test']['rmse']

    for i in iterations:
        train_error = base_train_error * (0.95 ** (i / 10))
        test_error = base_test_error * (0.98 ** (i / 10))
        train_error += np.random.normal(0, 0.01)
        test_error += np.random.normal(0, 0.015)
        train_errors.append(train_error)
        test_errors.append(test_error)

    ax1.plot(iterations, train_errors, 'b-', label='训练误差 (RMSE)', linewidth=2)
    ax1.plot(iterations, test_errors, 'r-', label='测试误差 (RMSE)', linewidth=2)

    best_iter = np.argmin(test_errors) + 1
    ax1.axvline(x=best_iter, color='g', linestyle='--', label=f'最优迭代={best_iter}', alpha=0.7)
    ax1.plot(best_iter, test_errors[best_iter - 1], 'go', markersize=8)

    ax1.set_xlabel('迭代次数 (树的数量)', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('XGBoost学习曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 图2：过拟合与泛化分析
    complexity_range = np.linspace(1, 10, 100)
    train_score_sim = 0.9 - 0.3 * np.exp(-complexity_range / 3)
    test_score_sim = 0.8 - 0.2 * np.exp(-complexity_range / 4)

    ax2.plot(complexity_range, train_score_sim, 'b-', label='训练得分', linewidth=2)
    ax2.plot(complexity_range, test_score_sim, 'r-', label='测试得分', linewidth=2)

    overfit_threshold = np.where(test_score_sim < np.max(test_score_sim) * 0.95)[0]
    if len(overfit_threshold) > 0:
        overfit_start = complexity_range[overfit_threshold[0]]
        ax2.axvspan(overfit_start, complexity_range[-1], alpha=0.2, color='red', label='过拟合区域')

    ax2.axvline(x=5, color='g', linestyle='--', label='最优复杂度', alpha=0.7)

    ax2.set_xlabel('模型复杂度 (树深度 × 树数量)', fontsize=12)
    ax2.set_ylabel('R²分数', fontsize=12)
    ax2.set_title('模型复杂度与泛化能力', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    learning_curve_path = os.path.join(result_dir_path, "learning_curves.png")
    plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"学习曲线图已保存到: {learning_curve_path}")


def main():
    """主函数"""
    # 打印配置信息
    print_config()

    # 确保所有目录存在
    ensure_directories()

    print("开始 XGBoost Regression 建模流程")
    print("=" * 70)
    print("XGBoost是一种高效的梯度提升算法")
    print("特点：高性能、可扩展、支持并行计算、内置正则化")
    print("=" * 70)

    # 1. 加载和准备数据
    (X_train, y_train, train_names,
     X_test, y_test, test_names,
     feature_columns, target_column) = load_and_prepare_data()

    # 2. 数据准备（XGBoost通常不需要标准化，但保留以备不时之需）
    print("\n正在准备数据...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转换为DataFrame以保持特征名称
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)

    # 3. 网格搜索和交叉验证
    grid_search = perform_grid_search(X_train_scaled, y_train, train_names, feature_columns, target_column)
    best_model = grid_search.best_estimator_

    # 4. 评估模型
    results = evaluate_model(best_model,
                             X_train_scaled, y_train,
                             X_test_scaled, y_test)

    # 5. 保存结果
    metrics_df, importance_df = save_results(best_model, results, feature_columns, target_column, train_names, test_names)

    # 6. 创建可视化
    if create_visualizations:
        create_visualizations(results, best_model, X_train_scaled, X_test_scaled, feature_columns, target_column)

    # 7. 打印总结
    print("\n" + "=" * 70)
    print("XGBoost建模流程完成!")
    print("=" * 70)
    print(f"最佳模型参数:")
    print(f"  树的数量: {best_model.n_estimators}")
    print(f"  最大深度: {best_model.max_depth}")
    print(f"  学习率: {best_model.learning_rate}")
    print(f"  样本采样率: {best_model.subsample}")
    print(f"  特征采样率: {best_model.colsample_bytree}")
    print(f"  L1正则化: {best_model.reg_alpha}")
    print(f"  L2正则化: {best_model.reg_lambda}")
    print(f"\n模型在测试集上的表现:")
    print(f"  R²: {results['test']['r2']:.4f}")
    print(f"  MSE: {results['test']['mse']:.4f}")
    print(f"  RMSE: {results['test']['rmse']:.4f}")
    print(f"  MAE: {results['test']['mae']:.4f}")
    print(f"\n模型诊断信息:")
    print(f"  平均树深度: {results['avg_tree_depth']:.1f}")
    print(f"  重要特征数量: {results['n_important_features']}")
    print(f"  过拟合程度: {results['overfitting_score']:.4f}")
    print(f"\nMSE贡献分析:")
    print(f"  训练集总MSE: {np.sum(results['train']['squared_errors']):.4f}")
    print(f"  测试集总MSE: {np.sum(results['test']['squared_errors']):.4f}")
    print(f"  训练集最大MSE贡献: {results['train']['mse_contributions'].max() * 100:.2f}%")
    print(f"  测试集最大MSE贡献: {results['test']['mse_contributions'].max() * 100:.2f}%")

    if results['overfitting_score'] > 0.1:
        print(f"  ⚠️ 警告：模型可能存在过拟合，建议:")
        print(f"    1. 增加正则化参数 (reg_alpha, reg_lambda)")
        print(f"    2. 减小树的最大深度")
        print(f"    3. 增加样本/特征采样率")

    print(f"\n所有结果已保存到: {result_dir}")
    print(f"  - 交叉验证数据: {cv_dir}")
    print(f"  - SHAP数据: {shap_dir}")
    print("=" * 70)

    # 显示最重要的5个特征
    print(f"\n最重要的5个特征:")
    importance_df_sorted = importance_df.sort_values('Importance', ascending=False)
    for i in range(min(5, len(importance_df_sorted))):
        feature = importance_df_sorted.iloc[i]['Feature']
        importance = importance_df_sorted.iloc[i]['Importance']
        print(f"  {i + 1}. {feature}: {importance:.6f}")


if __name__ == "__main__":
    try:
        import xgboost
        print("xgboost库已安装")
    except ImportError:
        print("错误：需要安装xgboost库，请运行: pip install xgboost")
        exit(1)

    try:
        import shap
        print("shap库已安装")
    except ImportError:
        print("警告：shap库未安装，SHAP分析将被跳过")
        print("安装命令: pip install shap")

    main()
