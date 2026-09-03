import os
import numpy as np
import pandas as pd


class ShapDataExporter:
    """SHAP数据导出器类"""

    def __init__(self, result_dir):
        """
        初始化SHAP数据导出器

        Parameters:
        -----------
        result_dir : str
            结果保存的主目录
        """
        self.result_dir = result_dir
        self.shap_data_dir = os.path.join(result_dir, "shap_data")

    def ensure_directory(self):
        """确保SHAP数据目录存在"""
        os.makedirs(self.shap_data_dir, exist_ok=True)
        return self.shap_data_dir

    def save_shap_values(self, shap_values, feature_columns):
        """
        保存SHAP值矩阵

        Parameters:
        -----------
        shap_values : numpy.ndarray
            SHAP值矩阵
        feature_columns : list
            特征名称列表
        """
        shap_values_df = pd.DataFrame(
            shap_values,
            columns=feature_columns
        )
        filepath = os.path.join(self.shap_data_dir, "shap_values.csv")
        shap_values_df.to_csv(filepath, index=False)
        print(f"  ✓ SHAP值矩阵已保存: {filepath}")
        return filepath

    def save_sample_features(self, X_sample, feature_columns):
        """
        保存用于SHAP分析的样本特征数据

        Parameters:
        -----------
        X_sample : pandas.DataFrame or numpy.ndarray
            样本特征数据
        feature_columns : list
            特征名称列表
        """
        if isinstance(X_sample, pd.DataFrame):
            X_sample_df = X_sample.copy()
        else:
            X_sample_df = pd.DataFrame(X_sample, columns=feature_columns)

        filepath = os.path.join(self.shap_data_dir, "shap_sample_features.csv")
        X_sample_df.to_csv(filepath, index=False)
        print(f"  ✓ SHAP样本特征已保存: {filepath}")
        return filepath, X_sample_df

    def save_feature_importance(self, shap_values, feature_columns):
        """
        保存SHAP特征重要性

        Parameters:
        -----------
        shap_values : numpy.ndarray
            SHAP值矩阵
        feature_columns : list
            特征名称列表
        """
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Mean_Abs_SHAP': mean_abs_shap,
            'SHAP_Rank': np.argsort(np.argsort(-mean_abs_shap)) + 1,
            'Normalized_Importance': mean_abs_shap / mean_abs_shap.sum()
        }).sort_values('Mean_Abs_SHAP', ascending=False)

        filepath = os.path.join(self.shap_data_dir, "shap_feature_importance.csv")
        shap_importance_df.to_csv(filepath, index=False)
        print(f"  ✓ SHAP特征重要性已保存: {filepath}")
        return filepath, shap_importance_df

    def save_sample_contributions(self, shap_values, X_sample_df, feature_columns, max_samples=10):
        """
        保存样本SHAP贡献详细数据

        Parameters:
        -----------
        shap_values : numpy.ndarray
            SHAP值矩阵
        X_sample_df : pandas.DataFrame
            样本特征数据
        feature_columns : list
            特征名称列表
        max_samples : int
            最大保存样本数
        """
        sample_contributions = []
        n_samples = min(max_samples, shap_values.shape[0])

        for i in range(n_samples):
            for j, feature in enumerate(feature_columns):
                sample_contributions.append({
                    'Sample_Index': i,
                    'Feature': feature,
                    'Feature_Value': X_sample_df.iloc[i, j],
                    'SHAP_Contribution': shap_values[i, j]
                })

        if sample_contributions:
            sample_contrib_df = pd.DataFrame(sample_contributions)
            filepath = os.path.join(self.shap_data_dir, "sample_shap_contributions.csv")
            sample_contrib_df.to_csv(filepath, index=False)
            print(f"  ✓ 样本SHAP贡献数据已保存: {filepath}")
            return filepath
        return None

    def save_base_value(self, expected_value):
        """
        保存SHAP基础值

        Parameters:
        -----------
        expected_value : float or array
            SHAP解释器的期望值
        """
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0] if len(expected_value) > 0 else expected_value

        base_value_df = pd.DataFrame({
            'Metric': ['Expected_Value'],
            'Value': [expected_value]
        })

        filepath = os.path.join(self.shap_data_dir, "shap_base_value.csv")
        base_value_df.to_csv(filepath, index=False)
        print(f"  ✓ SHAP基础值已保存: {filepath}")
        return filepath

    def save_summary_statistics(self, shap_values, feature_columns):
        """
        保存SHAP摘要统计信息

        Parameters:
        -----------
        shap_values : numpy.ndarray
            SHAP值矩阵
        feature_columns : list
            特征名称列表
        """
        shap_summary_stats = []
        for j, feature in enumerate(feature_columns):
            shap_summary_stats.append({
                'Feature': feature,
                'Mean_SHAP': np.mean(shap_values[:, j]),
                'Std_SHAP': np.std(shap_values[:, j]),
                'Min_SHAP': np.min(shap_values[:, j]),
                'Max_SHAP': np.max(shap_values[:, j]),
                'Positive_Effect_Ratio': np.mean(shap_values[:, j] > 0),
                'Negative_Effect_Ratio': np.mean(shap_values[:, j] < 0)
            })

        shap_stats_df = pd.DataFrame(shap_summary_stats)
        filepath = os.path.join(self.shap_data_dir, "shap_summary_statistics.csv")
        shap_stats_df.to_csv(filepath, index=False)
        print(f"  ✓ SHAP统计信息已保存: {filepath}")
        return filepath

    def save_feature_interactions(self, shap_importance_df, X_sample_df, feature_columns, top_n=10):
        """
        保存特征交互数据

        Parameters:
        -----------
        shap_importance_df : pandas.DataFrame
            SHAP特征重要性数据框
        X_sample_df : pandas.DataFrame
            样本特征数据
        feature_columns : list
            特征名称列表
        top_n : int
            考虑的特征数量
        """
        try:
            top_n_features = min(top_n, len(feature_columns))
            top_features = shap_importance_df['Feature'].head(top_n_features).tolist()
            top_feature_indices = [feature_columns.index(f) for f in top_features]

            interaction_data = []
            for i, idx1 in enumerate(top_feature_indices):
                for j, idx2 in enumerate(top_feature_indices[i + 1:], i + 1):
                    feature1_vals = X_sample_df.iloc[:, idx1].values
                    feature2_vals = X_sample_df.iloc[:, idx2].values

                    correlation = np.corrcoef(feature1_vals, feature2_vals)[0, 1] if len(feature1_vals) > 1 else 0

                    interaction_data.append({
                        'Feature_1': feature_columns[idx1],
                        'Feature_2': feature_columns[idx2],
                        'Correlation': correlation,
                        'Abs_Correlation': abs(correlation)
                    })

            if interaction_data:
                interaction_df = pd.DataFrame(interaction_data)
                interaction_df = interaction_df.sort_values('Abs_Correlation', ascending=False)
                filepath = os.path.join(self.shap_data_dir, "feature_interactions.csv")
                interaction_df.to_csv(filepath, index=False)
                print(f"  ✓ 特征交互数据已保存: {filepath}")
                return filepath
        except Exception as e:
            print(f"  ⚠️ 特征交互数据保存失败: {e}")
        return None

    def save_quantiles(self, shap_values, feature_columns):
        """
        保存SHAP值分位数信息

        Parameters:
        -----------
        shap_values : numpy.ndarray
            SHAP值矩阵
        feature_columns : list
            特征名称列表
        """
        shap_quantiles = []
        for j, feature in enumerate(feature_columns):
            quantiles = np.percentile(shap_values[:, j], [0, 25, 50, 75, 100])
            shap_quantiles.append({
                'Feature': feature,
                'Q0_Min': quantiles[0],
                'Q25': quantiles[1],
                'Q50_Median': quantiles[2],
                'Q75': quantiles[3],
                'Q100_Max': quantiles[4],
                'IQR': quantiles[3] - quantiles[1]
            })

        shap_quantiles_df = pd.DataFrame(shap_quantiles)
        filepath = os.path.join(self.shap_data_dir, "shap_quantiles.csv")
        shap_quantiles_df.to_csv(filepath, index=False)
        print(f"  ✓ SHAP分位数数据已保存: {filepath}")
        return filepath

    def export_all(self, explainer, shap_values, X_sample, feature_columns, max_samples=10, top_n_interactions=10):
        """
        导出所有SHAP数据

        Parameters:
        -----------
        explainer : shap.TreeExplainer
            SHAP解释器对象
        shap_values : numpy.ndarray
            SHAP值矩阵
        X_sample : pandas.DataFrame or numpy.ndarray
            样本特征数据
        feature_columns : list
            特征名称列表
        max_samples : int
            最大保存样本数
        top_n_interactions : int
            考虑交互的特征数量

        Returns:
        --------
        str : SHAP数据保存目录
        """
        print("\n" + "=" * 50)
        print("开始导出SHAP数据")
        print("=" * 50)

        # 确保目录存在
        self.ensure_directory()

        # 保存SHAP值矩阵
        self.save_shap_values(shap_values, feature_columns)

        # 保存样本特征
        X_sample_path, X_sample_df = self.save_sample_features(X_sample, feature_columns)

        # 保存特征重要性
        importance_path, shap_importance_df = self.save_feature_importance(shap_values, feature_columns)

        # 保存样本贡献
        self.save_sample_contributions(shap_values, X_sample_df, feature_columns, max_samples)

        # 保存基础值
        self.save_base_value(explainer.expected_value)

        # 保存统计信息
        self.save_summary_statistics(shap_values, feature_columns)

        # 保存特征交互
        self.save_feature_interactions(shap_importance_df, X_sample_df, feature_columns, top_n_interactions)

        # 保存分位数
        self.save_quantiles(shap_values, feature_columns)

        print("=" * 50)
        print(f"SHAP数据导出完成！保存目录: {self.shap_data_dir}")
        print("=" * 50)

        return self.shap_data_dir


# 便捷函数，用于直接调用
def export_shap_data(explainer, shap_values, X_sample, feature_columns, result_dir,
                     max_samples=10, top_n_interactions=10):
    """
    便捷函数：导出SHAP数据

    Parameters:
    -----------
    explainer : shap.TreeExplainer
        SHAP解释器对象
    shap_values : numpy.ndarray
        SHAP值矩阵
    X_sample : pandas.DataFrame or numpy.ndarray
        样本特征数据
    feature_columns : list
        特征名称列表
    result_dir : str
        结果保存的主目录
    max_samples : int
        最大保存样本数
    top_n_interactions : int
        考虑交互的特征数量

    Returns:
    --------
    str : SHAP数据保存目录
    """
    exporter = ShapDataExporter(result_dir)
    return exporter.export_all(explainer, shap_values, X_sample, feature_columns,
                               max_samples, top_n_interactions)


if __name__ == "__main__":
    # 测试代码
    print("SHAP数据导出模块")
    print("此模块用于导出SHAP分析数据，不应直接运行")
    print("请在主程序中导入并使用：")
    print("  from shap_data_exporter import export_shap_data")
    print("  export_shap_data(explainer, shap_values, X_sample, feature_columns, result_dir)")