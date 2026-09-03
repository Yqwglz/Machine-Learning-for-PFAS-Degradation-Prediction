import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
import chardet  # 添加编码检测库

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
# 设置路径
data_dir = Path("LOGO/LOCO/multi/999")
output_dir = Path("L-LOGO/LOCO/5/999")
output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录

train_file = "train_standardized_no_groups.csv"

# 读取数据
file_path = data_dir / train_file
print(f"正在读取文件: {file_path}")


# ===== 添加特殊字符处理功能开始 =====
def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def clean_column_names(columns):
    """清洗列名中的特殊字符"""
    cleaned_columns = []
    for col in columns:
        # 去除首尾空格
        col = col.strip()
        # 替换常见的特殊字符
        col = col.replace('(', '_').replace(')', '_')
        col = col.replace('[', '_').replace(']', '_')
        col = col.replace('{', '_').replace('}', '_')
        col = col.replace('/', '_').replace('\\', '_')
        col = col.replace(' ', '_')
        col = col.replace('-', '_')
        col = col.replace('+', '_')
        col = col.replace('*', '_')
        col = col.replace('&', '_')
        col = col.replace('%', '_')
        col = col.replace('$', '_')
        col = col.replace('#', '_')
        col = col.replace('@', '_')
        col = col.replace('!', '_')
        col = col.replace('?', '_')
        col = col.replace(':', '_')
        col = col.replace(';', '_')
        col = col.replace(',', '_')
        col = col.replace('.', '_')
        col = col.replace("'", '_')
        col = col.replace('"', '_')
        # 去除连续的下划线
        while '__' in col:
            col = col.replace('__', '_')
        # 去除首尾下划线
        col = col.strip('_')
        cleaned_columns.append(col)
    return cleaned_columns


# 尝试多种编码读取
encodings_to_try = ['utf-8', 'gbk', ]

df = None
for encoding in encodings_to_try:
    try:
        print(f"尝试使用编码: {encoding}")
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"成功使用编码: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"编码 {encoding} 失败")
        continue
    except Exception as e:
        print(f"使用编码 {encoding} 时出错: {e}")
        continue

# ===== 特殊字符处理功能结束 =====

print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")


target_col = None
for col in df.columns:
    if 'Kobs' in col.lower():
        target_col = col
        break

# 寻找名称列（可能是Name或Name的变体）
name_col = None
for col in df.columns:
    if col.lower() == 'name':
        name_col = col
        break

if name_col is not None and target_col is not None:
    X = df.drop([name_col, target_col], axis=1)
    y = df[target_col]
    feature_names = X.columns.tolist()
elif target_col is not None:
    # 如果没找到Name列，但有目标列
    y = df[target_col]
    X = df.drop([target_col], axis=1)
    if name_col is not None:
        X = X.drop([name_col], axis=1)
    feature_names = X.columns.tolist()
else:
    # 如果列名不完全匹配，尝试自动识别
    print("未找到'Name'或'Kobs'列，尝试自动识别特征和目标变量...")
    # 假设最后一列是目标变量
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    # 如果第一列是Name，排除它
    if df.columns[0].lower() == 'name':
        X = X.iloc[:, 1:]
    feature_names = X.columns.tolist()

print(f"特征数量: {len(feature_names)}")
print(f"样本数量: {len(y)}")



scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 使用LassoCV进行特征选择（交叉验证选择最佳alpha）
print("\n正在进行Lasso回归特征选择...")
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_scaled, y)

# 获取选择后的特征
selected_features_mask = lasso_cv.coef_ != 0
selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_features_mask[i]]
selected_coefs = lasso_cv.coef_[selected_features_mask]

print(f"\n最佳alpha值: {lasso_cv.alpha_:.6f}")
print(f"原始特征数量: {len(feature_names)}")
print(f"选择后特征数量: {len(selected_features)}")
print(f"特征选择比例: {len(selected_features) / len(feature_names) * 100:.2f}%")

# 输出选择的特征和系数
print("\n选择的特征及其系数:")
for feature, coef in zip(selected_features, selected_coefs):
    print(f"{feature}: {coef:.6f}")

# 保存选择后的特征数据到CSV文件
selected_features_file = output_dir / "selected_features.csv"
selected_features_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': selected_coefs,
    'Absolute_Coefficient': np.abs(selected_coefs)
}).sort_values('Absolute_Coefficient', ascending=False)

selected_features_df.to_csv(selected_features_file, index=False, encoding='utf-8-sig')
print(f"\n已保存选择的特征到: {selected_features_file}")

# 保存包含选择特征的完整数据集
if 'Name' in df.columns:
    selected_X = df[['Name'] + selected_features + ['Kobs']]
else:
    selected_X = df[selected_features + ['Kobs']]

selected_data_file = output_dir / "train_des.csv"
selected_X.to_csv(selected_data_file, index=False, encoding='utf-8-sig')
print(f"已保存包含选择特征的数据集到: {selected_data_file}")

# 创建可视化图表
plt.figure(figsize=(15, 10))

# 1. 特征系数图
plt.subplot(2, 2, 1)
sorted_indices = np.argsort(np.abs(lasso_cv.coef_))[::-1]
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_coefs = lasso_cv.coef_[sorted_indices]

colors = ['red' if coef == 0 else 'blue' for coef in sorted_coefs]
bars = plt.bar(range(len(sorted_features)), sorted_coefs, color=colors)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('特征')
plt.ylabel('系数值')
plt.title('Lasso回归特征系数')
plt.xticks(ticks=range(len(sorted_features)), labels=sorted_features, rotation=90, fontsize=8)
plt.grid(True, alpha=0.3)

# 2. 系数绝对值排序图
plt.subplot(2, 2, 2)
abs_coefs = np.abs(lasso_cv.coef_[sorted_indices])
plt.bar(range(len(sorted_features)), abs_coefs, color='green', alpha=0.7)
plt.xlabel('特征')
plt.ylabel('系数绝对值')
plt.title('特征系数绝对值排序')
plt.xticks(ticks=range(len(sorted_features)), labels=sorted_features, rotation=90, fontsize=8)
plt.grid(True, alpha=0.3)

# 3. 选择特征与原始特征数量对比
plt.subplot(2, 2, 3)
labels = ['原始特征', '选择特征']
counts = [len(feature_names), len(selected_features)]
colors = ['lightblue', 'lightcoral']
plt.bar(labels, counts, color=colors)
plt.ylabel('数量')
plt.title('特征选择前后数量对比')
for i, count in enumerate(counts):
    plt.text(i, count + max(counts) * 0.02, str(count), ha='center', fontweight='bold')

# 4. 交叉验证路径图
plt.subplot(2, 2, 4)
alphas = lasso_cv.alphas_
mse_path = lasso_cv.mse_path_
mean_mse = mse_path.mean(axis=1)
std_mse = mse_path.std(axis=1)
plt.semilogx(alphas, mean_mse, color='black', lw=2, label='Mean MSE')
plt.fill_between(alphas, mean_mse - std_mse, mean_mse + std_mse, alpha=0.2, color='gray', label='MSE ±1 Std Dev')
plt.axvline(lasso_cv.alpha_, color='red', linestyle='--', label=f'Best alpha: {lasso_cv.alpha_:.6f}')
plt.xlabel('Log alpha')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()

plt.tight_layout()

# 保存图表
chart_file = output_dir / "lasso_feature_selection.png"
plt.savefig(chart_file, dpi=300, bbox_inches='tight')
print(f"已保存图表到: {chart_file}")

# 创建单独的系数重要性图
plt.figure(figsize=(12, 6))
if len(selected_features) > 0:
    selected_indices = [feature_names.index(f) for f in selected_features]
    selected_sorted = sorted(zip(selected_features, selected_coefs), key=lambda x: abs(x[1]), reverse=True)
    selected_features_sorted = [item[0] for item in selected_sorted]
    selected_coefs_sorted = [item[1] for item in selected_sorted]

    colors = ['red' if coef < 0 else 'blue' for coef in selected_coefs_sorted]
    plt.barh(range(len(selected_features_sorted)), selected_coefs_sorted, color=colors)
    plt.yticks(range(len(selected_features_sorted)), selected_features_sorted)
    plt.xlabel('系数值')
    plt.title('选择特征的系数值（按绝对值排序）')
    plt.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, v in enumerate(selected_coefs_sorted):
        plt.text(v, i, f' {v:.4f}', va='center', fontsize=9,
                 color='black' if abs(v) < max(abs(np.array(selected_coefs_sorted))) * 0.5 else 'white')
else:
    plt.text(0.5, 0.5, '没有特征被选择\n所有系数都被压缩为零',
             ha='center', va='center', fontsize=14)

plt.tight_layout()
coef_chart_file = output_dir / "selected_features_coefficients.png"
plt.savefig(coef_chart_file, dpi=300, bbox_inches='tight')
print(f"已保存系数图到: {coef_chart_file}")

print("\n" + "=" * 60)
print("Lasso特征选择完成!")
print(f"选择特征数: {len(selected_features)}")
print(f"输出文件保存至: {output_dir}")
print("=" * 60)

# 显示图表（可选）
# plt.show()