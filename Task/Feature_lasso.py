import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import warnings
import chardet
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FILE_PATH = "../Data/Splits/859/train_859_standardized.csv"
OUTPUT_DIR = '../Data/Lasso/859'
TARGET_COLUMN = "Kobs"
NAME_COLUMN = "Name"
RANDOM_STATE = 42
CV_FOLDS = 3
MAX_ITER = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 从输入文件路径自动生成输出文件名
input_filename = os.path.basename(FILE_PATH)
input_stem = os.path.splitext(input_filename)[0]  # 去掉.csv扩展名

# 移除标准化后缀（如果存在）
if '_standardized' in input_stem:
    output_stem = input_stem.replace('_standardized', '')
elif '_raw' in input_stem:
    output_stem = input_stem.replace('_raw', '')
else:
    output_stem = input_stem

# 生成输出文件名
selected_features_filename = f"{output_stem}_selected_features.csv"
selected_data_filename = f"{output_stem}.csv"
chart_filename = f"{output_stem}_lasso_cv_path.png"

print(f"Input file: {FILE_PATH}")
print(f"Output stem: {output_stem}")
print(f"Output files:")
print(f"  - {selected_features_filename}")
print(f"  - {selected_data_filename}")
print(f"  - {chart_filename}")


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']


print(f"\nReading file: {FILE_PATH}")

encodings_to_try = ['utf-8', 'gbk']

df = None
for encoding in encodings_to_try:
    try:
        print(f"Trying encoding: {encoding}")
        df = pd.read_csv(FILE_PATH, encoding=encoding)
        print(f"Successfully loaded with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Encoding {encoding} failed")
        continue
    except Exception as e:
        print(f"Error with encoding {encoding}: {e}")
        continue

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

target_col = None
for col in df.columns:
    if TARGET_COLUMN.lower() in col.lower():
        target_col = col
        break

name_col = None
for col in df.columns:
    if col.lower() == NAME_COLUMN.lower():
        name_col = col
        break

if name_col is not None and target_col is not None:
    X = df.drop([name_col, target_col], axis=1)
    y = df[target_col]
    feature_names = X.columns.tolist()
elif target_col is not None:
    y = df[target_col]
    X = df.drop([target_col], axis=1)
    if name_col is not None:
        X = X.drop([name_col], axis=1)
    feature_names = X.columns.tolist()
else:
    print("Target column not found, attempting automatic identification...")
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    if df.columns[0].lower() == NAME_COLUMN.lower():
        X = X.iloc[:, 1:]
    feature_names = X.columns.tolist()

print(f"Number of features: {len(feature_names)}")
print(f"Number of samples: {len(y)}")

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

print("\nPerforming Lasso regression feature selection...")
lasso_cv = LassoCV(cv=CV_FOLDS, random_state=RANDOM_STATE, max_iter=MAX_ITER)
lasso_cv.fit(X_scaled, y)

selected_features_mask = lasso_cv.coef_ != 0
selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_features_mask[i]]
selected_coefs = lasso_cv.coef_[selected_features_mask]

print(f"\nBest alpha value: {lasso_cv.alpha_:.6f}")
print(f"Original feature count: {len(feature_names)}")
print(f"Selected feature count: {len(selected_features)}")
print(f"Feature selection ratio: {len(selected_features) / len(feature_names) * 100:.2f}%")

print("\nSelected features and their coefficients:")
for feature, coef in zip(selected_features, selected_coefs):
    print(f"{feature}: {coef:.6f}")

selected_features_file = os.path.join(OUTPUT_DIR, selected_features_filename)
selected_features_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': selected_coefs,
    'Absolute_Coefficient': np.abs(selected_coefs)
}).sort_values('Absolute_Coefficient', ascending=False)

selected_features_df.to_csv(selected_features_file, index=False, encoding='utf-8-sig')
print(f"\nSelected features saved to: {selected_features_file}")

if name_col in df.columns:
    selected_X = df[[name_col] + selected_features + [target_col]]
else:
    selected_X = df[selected_features + [target_col]]

selected_data_file = os.path.join(OUTPUT_DIR, selected_data_filename)
selected_X.to_csv(selected_data_file, index=False, encoding='utf-8-sig')
print(f"Selected feature dataset saved to: {selected_data_file}")

plt.figure(figsize=(6, 5))

alphas = lasso_cv.alphas_
mse_path = lasso_cv.mse_path_
mean_mse = mse_path.mean(axis=1)
std_mse = mse_path.std(axis=1)

ax = plt.gca()

ax.minorticks_off()

def sci_notation(x, pos):
    exponent = int(np.log10(x))
    return f'$10^{{{exponent}}}$'

ax.xaxis.set_major_formatter(ticker.FuncFormatter(sci_notation))

plt.semilogx(alphas, mean_mse, color='black', lw=2, label='Mean MSE')
plt.fill_between(alphas, mean_mse - std_mse, mean_mse + std_mse,
                  alpha=0.2, color='gray', label='MSE +/- 1 Std Dev')
plt.axvline(lasso_cv.alpha_, color='red', linestyle='--',
            label=f'Best alpha: {lasso_cv.alpha_:.6f}')
plt.grid(False)

plt.xlabel('Log alpha', fontsize=12, fontweight='bold')
plt.ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')

plt.legend(frameon=False, prop={'size': 12, 'weight': 'bold'})

ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
for label in ax.get_xticklabels():
    label.set_weight('bold')
for label in ax.get_yticklabels():
    label.set_weight('bold')

plt.tight_layout()

chart_file = os.path.join(OUTPUT_DIR, chart_filename)
plt.savefig(chart_file, dpi=300, bbox_inches='tight')
print(f"Chart saved to: {chart_file}")

print("\n" + "=" * 60)
print("Lasso Feature Selection Completed!")
print(f"Selected feature count: {len(selected_features)}")
print(f"Output files saved to: {OUTPUT_DIR}")
print("=" * 60)