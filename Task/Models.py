import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import xgboost as xgb
import warnings
from pathlib import Path
import joblib
import shap

warnings.filterwarnings('ignore')

result_dir = "./result"

train_path = '../Data/train_des.csv'
test_path = '../Data/Original/test_original_standardized.csv'

FIXED_SEED = 42

target_column = "Kobs"
exclude_columns = ["Name", target_column]

random_seed = 42

cv_n_splits = 5
cv_shuffle = True


shap_sample_size = 100
shap_background_size = 100

mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 100), (50, 30, 20), (100, 50, 25), ],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],  
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.005, 0.01],
    'max_iter': [100, 500, 1000],
    'batch_size': [32, 64, 128],
    'early_stopping': [True],
}

mlp_model_params = {
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 50
}

rf_param_grid = {
    'n_estimators': [ 200, 300, 400],  
    'max_depth': [ 10, 15, 20, 30, None],  
    'min_samples_split': [1, 2, 4, 5, 10, 20],  
    'min_samples_leaf': [ 2, 4, 5, 10], 
    'max_features': ['sqrt', 'log2', None], 
    'bootstrap': [True, False]  
}

rf_model_params = {
    'random_state': 42,
    'n_jobs': -1,
}

en_param_grid = {
    'alpha': [0.001, 0.1, 0.15, 0.2],  
    'l1_ratio': [0.05, 0.1, 0.15, 0.2, 0.3, 0.5], 
    'max_iter': [1000, 5000],  
    'selection': ['cyclic', 'random']  
}

en_model_params = {
    'random_state': 42,
    'tol': 0.0001
}

ridge_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

ridge_model_params = {
    'random_state': 42
}

cat_param_grid = {
    'iterations': [100, 200, 500],  
    'depth': [1, 2, 4],  
    'learning_rate': [0.05, 0.1], 
    'l2_leaf_reg': [3, 5, 7, 9], 
    'random_strength': [0.5, 1, 2], 
    'bagging_temperature': [0, 0.5, 1], 
    'border_count': [32, 64], 
    'random_seed': [42]
}

cat_model_params = {
    'loss_function': 'RMSE',
    'verbose': False,
    'thread_count': -1,
}

svm_param_grid = {
    'svm__kernel': ['linear', 'rbf', 'poly'],  
    'svm__C': [0.1, 1, 10, 100],  
    'svm__epsilon': [0.01, 0.1, 0.2], 
    'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1], 
    'svm__degree': [2, 3, 4],  
    'svm__coef0': [0.0, 0.5, 1.0],  
}

xgb_param_grid = {
'n_estimators': [100, 200],  
'max_depth': [3, 5, 7],  
'learning_rate': [0.01, 0.05, 0.1, 0.2], 
'subsample': [0.5, 0.7, 1.0],  
'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  
'min_child_weight': [1, 3, 5],  
'gamma': [0, 0.1, 0.2],  
'reg_alpha': [0, 0.1, 1],  
'reg_lambda': [1, 1.5, 2], 
'random_state': [42]
}

xgb_model_params = {
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'verbosity': 0,
}

grid_search_params = {
    'scoring': 'neg_mean_squared_error',
    'n_jobs': -1,
    'verbose': 0,
    'refit': True
}

font_sans_serif = ['SimSun', 'Times New Roman']
plot_style = 'seaborn-v0_8-darkgrid'

plt.rcParams['font.sans-serif'] = font_sans_serif
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(random_seed)


class MultiModelComparison:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def perform_grid_search(self, model, param_grid, X_train, y_train, model_name):
        print(f"\nPerforming {model_name} grid search...")
        kfold = KFold(n_splits=cv_n_splits, shuffle=cv_shuffle, random_state=random_seed)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, **grid_search_params)
        grid_search.fit(X_train, y_train)
        print(f"{model_name} grid search completed!")
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search

    def perform_grid_search_catboost(self, model, param_grid, X_train, y_train, model_name):
        print(f"\nPerforming {model_name} grid search...")
        kfold = KFold(n_splits=cv_n_splits, shuffle=cv_shuffle, random_state=random_seed)
        cat_grid_search_params = grid_search_params.copy()
        cat_grid_search_params['n_jobs'] = 1
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, **cat_grid_search_params)
        grid_search.fit(X_train, y_train)
        print(f"{model_name} grid search completed!")
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search

    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name):
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

        overfitting_score = train_r2 - test_r2

        return {
            'model_name': model_name,
            'best_params': model.get_params(),
            'train': {'r2': train_r2, 'mse': train_mse, 'rmse': train_rmse, 'mae': train_mae,
                      'y_true': y_train, 'y_pred': y_train_pred},
            'test': {'r2': test_r2, 'mse': test_mse, 'rmse': test_rmse, 'mae': test_mae,
                     'y_true': y_test, 'y_pred': y_test_pred},
            'overfitting_score': overfitting_score
        }

    def print_comparison_table(self, all_results):
        print("\n" + "=" * 120)
        print("Model Performance Evaluation Results Comparison")
        print("=" * 120)

        header = f"{'Dataset':<12}"
        for result in all_results:
            model_name = result['model_name']
            if model_name == 'MLP':
                header += f"{'MLP':<12}"
            elif model_name == 'Random Forest':
                header += f"{'RF':<12}"
            elif model_name == 'ElasticNet':
                header += f"{'EN':<12}"
            elif model_name == 'Ridge':
                header += f"{'Ridge':<12}"
            elif model_name == 'CatBoost':
                header += f"{'CatBoost':<12}"
            elif model_name == 'SVM':
                header += f"{'SVM':<12}"
            elif model_name == 'XGBoost':
                header += f"{'XGBoost':<12}"
        print(header)
        print("-" * 120)

        train_r2_row = f"{'Train R2':<12}"
        for result in all_results:
            train_r2_row += f"{result['train']['r2']:<12.4f}"
        print(train_r2_row)

        test_r2_row = f"{'Test R2':<12}"
        for result in all_results:
            test_r2_row += f"{result['test']['r2']:<12.4f}"
        print(test_r2_row)

        overfit_row = f"{'Overfitting':<12}"
        for result in all_results:
            overfit_row += f"{result['overfitting_score']:<12.4f}"
        print(overfit_row)

        print("=" * 120)

    def create_combined_visualization(self, all_results, suffix=""):
        print("\nCreating visualization charts...")
        plt.style.use(plot_style)

        n_models = len(all_results)
        n_cols = min(4, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
        fig.suptitle('Multi-Model Performance Comparison', fontsize=18, y=1.02)

        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        colors = ['orange', 'purple', 'blue', 'green', 'brown', 'pink', 'cyan']

        for idx, (result, color) in enumerate(zip(all_results, colors[:n_models])):
            ax = axes[idx]

            y_train_true = result['train']['y_true']
            y_train_pred = result['train']['y_pred']
            ax.scatter(y_train_true, y_train_pred, alpha=0.6, color=color, s=50,
                       edgecolors='white', linewidth=0.5, label='Training set')

            y_test_true = result['test']['y_true']
            y_test_pred = result['test']['y_pred']
            ax.scatter(y_test_true, y_test_pred, alpha=0.6, color='red', s=80,
                       edgecolors='white', linewidth=0.5, label='Test set', marker='^')

            all_true = np.concatenate([y_train_true, y_test_true])
            all_pred = np.concatenate([y_train_pred, y_test_pred])
            min_val = min(all_true.min(), all_pred.min())
            max_val = max(all_true.max(), all_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, alpha=0.5)

            ax.set_xlabel('Actual values', fontsize=10)
            ax.set_ylabel('Predicted values', fontsize=10)
            ax.set_title(
                f'{result["model_name"]}\nTrain R2={result["train"]["r2"]:.4f}, Test R2={result["test"]["r2"]:.4f}',
                fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        viz_path = self.output_dir / f"models_comparison_r2{suffix}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined visualization saved to: {viz_path}")

    def compute_shap_values(self, model, X_train, feature_columns, model_dir, model_name):
        """计算并保存训练集SHAP值"""
        print(f"\n  Computing SHAP values for {model_name} (Training set only)...")

        shap_dir = model_dir / 'shap_data'
        shap_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 根据模型类型选择explainer
            if model_name in ['CatBoost', 'XGBoost', 'RandomForest']:
                # 树模型使用TreeExplainer
                print(f"  Using TreeExplainer for {model_name}")
                explainer = shap.TreeExplainer(model)
            else:
                # 非树模型使用KernelExplainer（MLP, SVM, ElasticNet, Ridge）
                print(f"  Using KernelExplainer for {model_name}")
                background_sample = min(shap_background_size, X_train.shape[0])
                background_data = X_train[:background_sample]

                if model_name == 'SVM':
                    def predict_func(X):
                        return model.predict(X)

                    explainer = shap.KernelExplainer(predict_func, background_data)
                else:
                    explainer = shap.KernelExplainer(model.predict, background_data)

            # 计算训练集SHAP值
            train_sample_size = min(shap_sample_size, X_train.shape[0])
            X_train_sample = X_train[:train_sample_size]
            print(f"  Computing SHAP values for {train_sample_size} training samples...")
            shap_values_train = explainer.shap_values(X_train_sample)

            # 保存SHAP值到CSV
            X_train_df = pd.DataFrame(X_train_sample, columns=feature_columns)

            # 训练集SHAP值
            shap_train_df = pd.DataFrame(shap_values_train, columns=feature_columns)
            shap_train_df['Sample_Index'] = range(1, len(shap_train_df) + 1)
            shap_train_df['Dataset'] = 'train'
            shap_train_path = shap_dir / 'shap_values_train.csv'
            shap_train_df.to_csv(shap_train_path, index=False, encoding='utf-8')
            print(f"  Training SHAP values saved to: {shap_train_path}")

            # 生成训练集SHAP可视化
            # 训练集特征重要性
            plt.figure(figsize=(14, max(8, len(feature_columns) * 0.3)))
            shap.summary_plot(shap_values_train, X_train_df, feature_names=feature_columns,
                              plot_type="bar", max_display=len(feature_columns), show=False)
            plt.title(f'SHAP Feature Importance (Train) - {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            shap_importance_train_path = shap_dir / 'shap_feature_importance_train.png'
            plt.savefig(shap_importance_train_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Training SHAP importance saved to: {shap_importance_train_path}")

            # 训练集summary plot
            plt.figure(figsize=(14, max(8, len(feature_columns) * 0.3)))
            shap.summary_plot(shap_values_train, X_train_df, feature_names=feature_columns,
                              max_display=len(feature_columns), show=False)
            plt.title(f'SHAP Summary Plot (Train) - {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            shap_summary_train_path = shap_dir / 'shap_summary_plot_train.png'
            plt.savefig(shap_summary_train_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Training SHAP summary saved to: {shap_summary_train_path}")

            return True

        except Exception as e:
            print(f"  SHAP computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_model_results(self, result, model, model_dir, X_train_scaled, feature_columns, model_name):
        model_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_dir / 'model.pkl'
        joblib.dump(model, model_file)

        train_df = pd.DataFrame({
            'dataset': 'train',
            'y_true': result['train']['y_true'],
            'y_pred': result['train']['y_pred']
        })

        test_df = pd.DataFrame({
            'dataset': 'test',
            'y_true': result['test']['y_true'],
            'y_pred': result['test']['y_pred']
        })

        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        predictions_file = model_dir / 'predictions.csv'
        combined_df.to_csv(predictions_file, index=False)

        log_file = model_dir / 'model_log.txt'
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Model: {result['model_name']}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Training data: {train_path}\n")
            f.write(f"Test data: {test_path}\n")
            f.write(f"Random seed: {FIXED_SEED}\n\n")

            f.write("-" * 80 + "\n")
            f.write("Best Model Parameters:\n")
            f.write("-" * 80 + "\n")
            for param, value in result['best_params'].items():
                f.write(f"  {param}: {value}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("Model Performance Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Training Set:\n")
            f.write(f"    R2 Score: {result['train']['r2']:.6f}\n")
            f.write(f"    MSE: {result['train']['mse']:.6f}\n")
            f.write(f"    RMSE: {result['train']['rmse']:.6f}\n")
            f.write(f"    MAE: {result['train']['mae']:.6f}\n")
            f.write(f"  Test Set:\n")
            f.write(f"    R2 Score: {result['test']['r2']:.6f}\n")
            f.write(f"    MSE: {result['test']['mse']:.6f}\n")
            f.write(f"    RMSE: {result['test']['rmse']:.6f}\n")
            f.write(f"    MAE: {result['test']['mae']:.6f}\n")
            f.write(f"  Overfitting Score: {result['overfitting_score']:.6f}\n")

        # 创建散点图
        plt.style.use(plot_style)
        fig, ax = plt.subplots(figsize=(8, 8))

        y_train_true = result['train']['y_true']
        y_train_pred = result['train']['y_pred']
        ax.scatter(y_train_true, y_train_pred, alpha=0.6, color='blue', s=50,
                   edgecolors='white', linewidth=0.5, label='Training set')

        y_test_true = result['test']['y_true']
        y_test_pred = result['test']['y_pred']
        ax.scatter(y_test_true, y_test_pred, alpha=0.6, color='red', s=80,
                   edgecolors='white', linewidth=0.5, label='Test set', marker='^')

        all_true = np.concatenate([y_train_true, y_test_true])
        all_pred = np.concatenate([y_train_pred, y_test_pred])
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, alpha=0.5)

        ax.set_xlabel('Actual values', fontsize=12)
        ax.set_ylabel('Predicted values', fontsize=12)
        ax.set_title(
            f'{result["model_name"]}\nTrain R2={result["train"]["r2"]:.4f}, Test R2={result["test"]["r2"]:.4f}',
            fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = model_dir / 'scatter_plot.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # 计算SHAP值（仅训练集）
        self.compute_shap_values(model, X_train_scaled, feature_columns, model_dir, model_name)

        print(f"Model results saved to: {model_dir}")


def main():
    output_dir = Path(result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("Multi-Model Comparison Pipeline with SHAP Analysis")
    print(f"{'=' * 80}")
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")
    print(f"Output directory: {result_dir}")
    print(f"Random seed: {FIXED_SEED}")
    print(f"{'=' * 80}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"\nTraining data loaded: {train_df.shape}")
    print(f"Test data loaded: {test_df.shape}")

    feature_columns = [col for col in train_df.columns if col not in exclude_columns]
    X_train = train_df[feature_columns].copy()
    y_train = train_df[target_column].copy()
    X_test = test_df[feature_columns].copy()
    y_test = test_df[target_column].copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")

    model_comparison = MultiModelComparison(output_dir)

    all_results = []

    # ElasticNet
    en = ElasticNet(**en_model_params)
    en_grid = model_comparison.perform_grid_search(en, en_param_grid, X_train_scaled, y_train, "ElasticNet")
    en_best = en_grid.best_estimator_
    en_results = model_comparison.evaluate_model(en_best, X_train_scaled, y_train, X_test_scaled, y_test, "ElasticNet")
    all_results.append(en_results)
    model_comparison.save_model_results(en_results, en_best, output_dir / 'ElasticNet', X_train_scaled, feature_columns,
                                        'ElasticNet')

    # Ridge
    ridge = Ridge(**ridge_model_params)
    ridge_grid = model_comparison.perform_grid_search(ridge, ridge_param_grid, X_train_scaled, y_train, "Ridge")
    ridge_best = ridge_grid.best_estimator_
    ridge_results = model_comparison.evaluate_model(ridge_best, X_train_scaled, y_train, X_test_scaled, y_test, "Ridge")
    all_results.append(ridge_results)
    model_comparison.save_model_results(ridge_results, ridge_best, output_dir / 'Ridge', X_train_scaled,
                                        feature_columns, 'Ridge')

    # MLP
    mlp = MLPRegressor(**mlp_model_params)
    mlp_grid = model_comparison.perform_grid_search(mlp, mlp_param_grid, X_train_scaled, y_train, "MLP")
    mlp_best = mlp_grid.best_estimator_
    mlp_results = model_comparison.evaluate_model(mlp_best, X_train_scaled, y_train, X_test_scaled, y_test, "MLP")
    all_results.append(mlp_results)
    model_comparison.save_model_results(mlp_results, mlp_best, output_dir / 'MLP', X_train_scaled, feature_columns,
                                        'MLP')

    # Random Forest
    rf = RandomForestRegressor(**rf_model_params)
    rf_grid = model_comparison.perform_grid_search(rf, rf_param_grid, X_train_scaled, y_train, "Random Forest")
    rf_best = rf_grid.best_estimator_
    rf_results = model_comparison.evaluate_model(rf_best, X_train_scaled, y_train, X_test_scaled, y_test,
                                                 "Random Forest")
    all_results.append(rf_results)
    model_comparison.save_model_results(rf_results, rf_best, output_dir / 'RandomForest', X_train_scaled,
                                        feature_columns, 'RandomForest')

    # CatBoost
    cat_model = CatBoostRegressor(**cat_model_params)
    cat_grid = model_comparison.perform_grid_search_catboost(cat_model, cat_param_grid, X_train_scaled, y_train,
                                                             "CatBoost")
    cat_best = cat_grid.best_estimator_
    cat_results = model_comparison.evaluate_model(cat_best, X_train_scaled, y_train, X_test_scaled, y_test, "CatBoost")
    all_results.append(cat_results)
    model_comparison.save_model_results(cat_results, cat_best, output_dir / 'CatBoost', X_train_scaled, feature_columns,
                                        'CatBoost')

    # SVM
    svm_pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVR())])
    svm_grid = model_comparison.perform_grid_search(svm_pipeline, svm_param_grid, X_train_scaled, y_train, "SVM")
    svm_best = svm_grid.best_estimator_
    svm_results = model_comparison.evaluate_model(svm_best, X_train_scaled, y_train, X_test_scaled, y_test, "SVM")
    all_results.append(svm_results)
    model_comparison.save_model_results(svm_results, svm_best, output_dir / 'SVM', X_train_scaled, feature_columns,
                                        'SVM')

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_model_params)
    xgb_grid = model_comparison.perform_grid_search(xgb_model, xgb_param_grid, X_train_scaled, y_train, "XGBoost")
    xgb_best = xgb_grid.best_estimator_
    xgb_results = model_comparison.evaluate_model(xgb_best, X_train_scaled, y_train, X_test_scaled, y_test, "XGBoost")
    all_results.append(xgb_results)
    model_comparison.save_model_results(xgb_results, xgb_best, output_dir / 'XGBoost', X_train_scaled, feature_columns,
                                        'XGBoost')

    model_comparison.print_comparison_table(all_results)
    model_comparison.create_combined_visualization(all_results, "_comparison")

    results_file = output_dir / 'model_results_summary.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("Model Training Results Summary\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Training data: {train_path}\n")
        f.write(f"Test data: {test_path}\n")
        f.write(f"Random seed: {FIXED_SEED}\n\n")
        f.write("Model performance:\n")
        for result in all_results:
            f.write(f"  {result['model_name']}:\n")
            f.write(f"    Train R2: {result['train']['r2']:.4f}\n")
            f.write(f"    Test R2: {result['test']['r2']:.4f}\n")
            f.write(f"    Train RMSE: {result['train']['rmse']:.4f}\n")
            f.write(f"    Test RMSE: {result['test']['rmse']:.4f}\n")
            f.write(f"    Train MAE: {result['train']['mae']:.4f}\n")
            f.write(f"    Test MAE: {result['test']['mae']:.4f}\n")
            f.write(f"    Overfitting: {result['overfitting_score']:.4f}\n\n")

    print(f"\n{'=' * 80}")
    print("Results saved to:", results_file)
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
