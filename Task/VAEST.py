import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
from datetime import datetime
import json
import random
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib

# ============================================================================
# 全局配置区域 - 所有可调参数都在这里
# ============================================================================

# 文件路径配置（直接在这里修改路径）
INPUT_FOLDER = "./lg_yuanshi_p/a/346"  # 输入文件夹路径（包含train_original253.csv和test253.csv）
OUTPUT_DIR = "./VAE_lg_yuanshi/process/346"  # 输出目录（可自由修改）

# 随机种子设置
RANDOM_SEED = 42

# VAE模型参数
VAE_PARAMS = {
    'hidden_dim': 256,  # 隐藏层维度
    'latent_dim': 64,  # 潜在空间维度
    'dropout_rate': 0.3,  # Dropout率
    'epochs': 1000,  # 训练轮次
    'batch_size': 64,  # 批大小
    'learning_rate': 1e-3,  # 学习率
    'weight_decay': 1e-5,  # 权重衰减
    'beta': 0.5,  # KL散度权重
    'gradient_clip': 1.0,  # 梯度裁剪阈值
}

# 数据生成参数
GENERATION_PARAMS = {
    'n_samples': 99,  # 生成样本数量
    'start_index': 1,  # Name列起始编号
    'output_prefix': 'VAE',  # 输出文件前缀
    'include_original': True,  # 是否在生成数据后追加原始数据
}

# 数据预处理参数
PREPROCESSING_PARAMS = {
    'normalize_categorical': True,  # 分类特征是否归一化到0-1
    'clip_percentile_low': 1,  # 数值特征裁剪下分位数
    'clip_percentile_high': 99,  # 数值特征裁剪上分位数
}

# 标准化配置
STANDARDIZATION_CONFIG = {
    'target_candidates': ['Kobs', 'kobs', 'K_obs', 'k_obs'],  # 目标列候选名称
    'name_col_index': 0,  # Name列索引（默认第一列）
    'non_na_ratio_threshold': 0.9,  # 非NA比例阈值（超过此比例认为是数值特征）
    'numeric_fill_strategy': 'mean',  # 数值特征填充策略: 'mean', 'median', 'zero'
    'categorical_fill_strategy': 'mode',  # 分类特征填充策略: 'mode', 'unknown'
    'handle_unknown': 'ignore',  # 处理未知类别: 'ignore', 'error'
    'sparse_output': False,  # 是否输出稀疏矩阵
    'use_all_categories': True,  # 是否使用训练集和测试集的所有类别
    'with_mean': True,  # 是否中心化（均值归零）
    'with_std': True,  # 是否缩放（标准差归1）
    'csv_encoding': 'utf-8-sig',  # CSV文件编码
}

# 输出配置
OUTPUT_CONFIG = {
    'save_vae_output': True,  # 保存VAE生成的数据
    'save_standardized_output': True,  # 保存标准化后的数据
    'save_log': True,  # 保存日志
}


# ============================================================================
# 自动生成路径（根据INPUT_FOLDER自动识别，不要手动修改）
# ============================================================================

def get_auto_paths():
    """根据输入文件夹自动生成所有路径"""
    # 获取文件夹名称
    folder_name = os.path.basename(os.path.abspath(INPUT_FOLDER))

    # 自动识别输入文件
    train_original_path = None
    test_path = None

    # 查找训练数据文件
    if os.path.exists(INPUT_FOLDER):
        files = os.listdir(INPUT_FOLDER)
        for file in files:
            if file.startswith('train_selected_original') and file.endswith('.csv'):
                train_original_path = os.path.join(INPUT_FOLDER, file)
                break

        # 查找测试数据文件
        for file in files:
            if file.startswith('test_original') and file.endswith('.csv'):
                test_path = os.path.join(INPUT_FOLDER, file)
                break

    # 如果自动查找失败，尝试常见的命名方式
    if train_original_path is None:
        train_original_path = os.path.join(INPUT_FOLDER, f'train_original{folder_name}.csv')

    if test_path is None:
        test_path = os.path.join(INPUT_FOLDER, f'test{folder_name}.csv')

    # 输出文件名
    vae_output_filename = f"{GENERATION_PARAMS['output_prefix']}_{folder_name}.csv"
    train_standardized_filename = f"train{GENERATION_PARAMS['output_prefix']}_{folder_name}.csv"
    test_standardized_filename = f"testVAE_{folder_name}.csv"

    return {
        'folder_name': folder_name,
        'input_folder': INPUT_FOLDER,
        'output_dir': OUTPUT_DIR,
        'train_original_path': train_original_path,
        'test_path': test_path,
        'vae_output_path': os.path.join(OUTPUT_DIR, vae_output_filename),
        'train_standardized_path': os.path.join(OUTPUT_DIR, train_standardized_filename),
        'test_standardized_path': os.path.join(OUTPUT_DIR, test_standardized_filename),
    }


# ============================================================================
# 以下代码保持不变
# ============================================================================

# 设置随机种子
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class VAEProcessor:
    """VAE数据生成处理器"""

    def __init__(self, paths):
        self.paths = paths
        self.numeric_features = []
        self.categorical_features = []
        self.numeric_scaler = None
        self.categorical_encoders = {}

    def detect_file_encoding(self, filepath):
        """检测文件编码"""
        try:
            import chardet
            with open(filepath, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                print(f"检测到编码: {encoding} (置信度: {confidence:.2%})")
                return encoding if confidence > 0.7 else 'utf-8'
        except:
            return 'utf-8'

    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        data_path = self.paths['train_original_path']

        if not os.path.exists(data_path):
            print(f"错误: 未找到数据文件 {data_path}")
            return None

        try:
            encoding = self.detect_file_encoding(data_path)
            df = pd.read_csv(data_path, encoding=encoding)
            print(f"数据加载成功，形状: {df.shape}")
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None

    def identify_columns(self, df):
        """识别所有特征列（排除Name列）"""
        print("\n识别特征列...")
        all_columns = df.columns.tolist()
        exclude_columns = ['Name']
        feature_columns = [col for col in all_columns if col not in exclude_columns]

        print(f"全部特征列 ({len(feature_columns)}个)")
        return feature_columns

    def preprocess_features(self, df, feature_columns):
        """预处理特征数据"""
        print(f"\n预处理特征数据 ({len(feature_columns)}个特征)...")

        numeric_features = []
        categorical_features = []

        for col in feature_columns:
            if col in df.columns:
                try:
                    if df[col].dtype in ['int64', 'float64']:
                        numeric_features.append(col)
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        numeric_features.append(col)
                    else:
                        try:
                            df[col] = pd.to_numeric(df[col])
                            numeric_features.append(col)
                        except:
                            categorical_features.append(col)
                except:
                    categorical_features.append(col)

        print(f"数值特征: {len(numeric_features)}个")
        print(f"分类特征: {len(categorical_features)}个")

        # 处理数值特征
        X_numeric = None
        if numeric_features:
            X_numeric_original = df[numeric_features].values
            self.numeric_scaler = StandardScaler()
            X_numeric = self.numeric_scaler.fit_transform(X_numeric_original)

        # 处理分类特征
        X_categorical = None
        if categorical_features:
            categorical_data_list = []
            for col in categorical_features:
                le = LabelEncoder()
                col_data = df[col].fillna('Missing').astype(str).values
                encoded = le.fit_transform(col_data).astype(np.float32)

                if len(le.classes_) > 1 and PREPROCESSING_PARAMS['normalize_categorical']:
                    encoded = encoded / (len(le.classes_) - 1)

                self.categorical_encoders[col] = le
                categorical_data_list.append(encoded.reshape(-1, 1))

            X_categorical = np.hstack(categorical_data_list)

        # 合并特征
        if X_numeric is not None and X_categorical is not None:
            X_features = np.hstack([X_numeric, X_categorical])
        elif X_numeric is not None:
            X_features = X_numeric
        elif X_categorical is not None:
            X_features = X_categorical
        else:
            return None

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        return X_features

    def train_vae(self, X_features):
        """训练VAE模型"""
        print("\n开始训练VAE模型...")
        print(
            f"训练参数: epochs={VAE_PARAMS['epochs']}, batch_size={VAE_PARAMS['batch_size']}, latent_dim={VAE_PARAMS['latent_dim']}")

        X_tensor = torch.FloatTensor(X_features)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=VAE_PARAMS['batch_size'], shuffle=True)

        input_dim = X_features.shape[1]
        model = CustomVAE(
            input_dim,
            hidden_dim=VAE_PARAMS['hidden_dim'],
            latent_dim=VAE_PARAMS['latent_dim'],
            dropout_rate=VAE_PARAMS['dropout_rate']
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=VAE_PARAMS['learning_rate'],
            weight_decay=VAE_PARAMS['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=VAE_PARAMS['epochs'])

        model.train()
        for epoch in range(VAE_PARAMS['epochs']):
            epoch_total_loss = 0
            n_batches = 0

            for batch_idx, (data,) in enumerate(dataloader):
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                total_loss, _, _ = vae_loss(recon_batch, data, mu, logvar, VAE_PARAMS['beta'])
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=VAE_PARAMS['gradient_clip'])
                optimizer.step()

                epoch_total_loss += total_loss.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 100 == 0 or epoch == 0:
                avg_loss = epoch_total_loss / n_batches
                print(f'轮次 [{epoch + 1:03d}/{VAE_PARAMS["epochs"]:03d}] - 损失: {avg_loss:.6f}')

        print("VAE模型训练完成!")
        return model

    def generate_samples(self, model, n_samples):
        """生成样本"""
        print(f"\n生成 {n_samples} 个样本...")

        model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, VAE_PARAMS['latent_dim'])
            generated_features = model.decode(z).cpu().numpy()

        return generated_features

    def postprocess_generated(self, generated_features):
        """后处理生成的特征"""
        n_numeric = len(self.numeric_features)
        n_categorical = len(self.categorical_features)

        generated_numeric = generated_features[:, :n_numeric]
        generated_categorical = generated_features[:, n_numeric:] if n_categorical > 0 else None

        # 反标准化数值特征
        if self.numeric_scaler is not None and n_numeric > 0:
            generated_numeric = self.numeric_scaler.inverse_transform(generated_numeric)

            for i in range(generated_numeric.shape[1]):
                col_data = generated_numeric[:, i]
                q1 = np.percentile(col_data, PREPROCESSING_PARAMS['clip_percentile_low'])
                q99 = np.percentile(col_data, PREPROCESSING_PARAMS['clip_percentile_high'])
                col_data[col_data < q1] = q1
                col_data[col_data > q99] = q99
                generated_numeric[:, i] = col_data

        # 处理分类特征
        categorical_data = {}
        if generated_categorical is not None and n_categorical > 0:
            for i, col in enumerate(self.categorical_features):
                if i < generated_categorical.shape[1]:
                    le = self.categorical_encoders[col]
                    cat_encoded = generated_categorical[:, i]

                    if len(le.classes_) > 1 and PREPROCESSING_PARAMS['normalize_categorical']:
                        cat_encoded_int = np.round(cat_encoded * (len(le.classes_) - 1))
                    else:
                        cat_encoded_int = np.round(cat_encoded)

                    cat_encoded_int = np.clip(cat_encoded_int, 0, len(le.classes_) - 1).astype(int)

                    try:
                        decoded_labels = le.inverse_transform(cat_encoded_int)
                    except:
                        decoded_labels = [le.classes_[min(val, len(le.classes_) - 1)] for val in cat_encoded_int]
                        decoded_labels = np.array(decoded_labels)

                    categorical_data[col] = decoded_labels

        return generated_numeric, categorical_data

    def create_dataframe(self, generated_numeric, categorical_data, n_samples):
        """创建DataFrame"""
        result_df = pd.DataFrame(generated_numeric, columns=self.numeric_features)

        for col in self.categorical_features:
            if col in categorical_data:
                result_df[col] = categorical_data[col]

        result_df.insert(0, 'Name', [f"{i}" for i in range(GENERATION_PARAMS['start_index'],
                                                           GENERATION_PARAMS['start_index'] + n_samples)])

        return result_df

    def combine_with_original(self, generated_df, original_df):
        """合并生成数据和原始数据"""
        if list(generated_df.columns) != list(original_df.columns):
            common_columns = [col for col in generated_df.columns if col in original_df.columns]
            generated_df = generated_df[common_columns]
            original_df = original_df[common_columns]

        if 'Name' in generated_df.columns and 'Name' in original_df.columns:
            original_names = original_df['Name'].astype(str)
            numeric_names = []
            for name in original_names:
                try:
                    numeric_names.append(int(name))
                except:
                    numeric_names.append(0)

            max_original_name = max(numeric_names) if numeric_names else 0
            start_index = max_original_name + 1
            generated_df['Name'] = [str(i) for i in range(start_index, start_index + len(generated_df))]

        combined_df = pd.concat([generated_df, original_df], ignore_index=True)
        return combined_df

    def run(self):
        """运行VAE处理流程"""
        print("=" * 80)
        print("VAE数据生成系统")
        print("=" * 80)
        print(f"输入文件夹: {self.paths['input_folder']}")
        print(f"训练数据: {self.paths['train_original_path']}")
        print(f"输出目录: {self.paths['output_dir']}")

        # 加载数据
        df = self.load_data()
        if df is None:
            return None

        # 识别特征
        feature_columns = self.identify_columns(df)

        # 预处理特征
        X_features = self.preprocess_features(df, feature_columns)
        if X_features is None:
            return None

        # 训练VAE
        model = self.train_vae(X_features)

        # 生成样本
        n_samples = GENERATION_PARAMS['n_samples']
        generated_features = self.generate_samples(model, n_samples)

        # 后处理
        generated_numeric, categorical_data = self.postprocess_generated(generated_features)

        # 创建DataFrame
        generated_df = self.create_dataframe(generated_numeric, categorical_data, n_samples)

        # 合并数据
        if GENERATION_PARAMS['include_original']:
            combined_df = self.combine_with_original(generated_df, df)
        else:
            combined_df = generated_df

        # 保存结果
        os.makedirs(self.paths['output_dir'], exist_ok=True)

        if OUTPUT_CONFIG['save_vae_output']:
            combined_df.to_csv(self.paths['vae_output_path'], index=False,
                               encoding=STANDARDIZATION_CONFIG['csv_encoding'])
            print(f"\n✓ VAE生成数据已保存: {self.paths['vae_output_path']}")

        return combined_df


class DataStandardizer:
    """数据标准化处理器"""

    def __init__(self, paths):
        self.paths = paths
        self.train_path = paths['vae_output_path']
        self.test_path = paths['test_path']
        self.output_dir = paths['output_dir']

        self.target_scaler = StandardScaler(
            with_mean=STANDARDIZATION_CONFIG['with_mean'],
            with_std=STANDARDIZATION_CONFIG['with_std']
        )
        self.preprocessor = None
        self.feature_cols = []
        self.numeric_features = []
        self.string_features = []
        self.target_col = None
        self.name_col = None
        self.train_raw = None
        self.test_raw = None
        self.train_processed = None
        self.test_processed = None
        self.feature_names_processed = []

    def load_data(self):
        """加载训练集和测试集"""
        print("\n" + "=" * 60)
        print("加载数据用于标准化")
        print("=" * 60)

        if self.train_path.endswith('.csv'):
            self.train_raw = pd.read_csv(self.train_path)
        else:
            self.train_raw = pd.read_excel(self.train_path)
        print(f"训练集加载成功: {self.train_path}")
        print(f"  形状: {self.train_raw.shape}")

        if self.test_path.endswith('.csv'):
            self.test_raw = pd.read_csv(self.test_path)
        else:
            self.test_raw = pd.read_excel(self.test_path)
        print(f"测试集加载成功: {self.test_path}")
        print(f"  形状: {self.test_raw.shape}")

        return self.train_raw, self.test_raw

    def identify_columns(self):
        """识别列类型"""
        print("\n识别列类型...")

        self.name_col = self.train_raw.columns[STANDARDIZATION_CONFIG['name_col_index']]

        self.target_col = None
        for col in STANDARDIZATION_CONFIG['target_candidates']:
            if col in self.train_raw.columns:
                self.target_col = col
                break

        if self.target_col is None:
            for col in self.train_raw.columns:
                if 'obs' in col.lower() or col.lower() == 'k':
                    self.target_col = col
                    break

        if self.target_col is None:
            self.target_col = self.train_raw.columns[-1]
            print(f"警告: 未找到Kobs列，使用最后一列 '{self.target_col}' 作为目标列")

        self.feature_cols = [col for col in self.train_raw.columns
                             if col not in [self.name_col, self.target_col]]

        self.numeric_features = []
        self.string_features = []

        for col in self.feature_cols:
            col_data = self.train_raw[col]
            try:
                numeric_test = pd.to_numeric(col_data, errors='coerce')
                non_na_ratio = numeric_test.notna().mean()

                if non_na_ratio > STANDARDIZATION_CONFIG['non_na_ratio_threshold']:
                    self.numeric_features.append(col)
                else:
                    self.string_features.append(col)
            except:
                self.string_features.append(col)

        print(f"数值特征: {len(self.numeric_features)}个")
        print(f"分类特征: {len(self.string_features)}个")

    def preprocess_features(self):
        """预处理特征"""
        print("\n特征预处理...")

        X_train = self.train_raw[self.feature_cols].copy()
        X_test = self.test_raw[self.feature_cols].copy()

        # 处理数值特征
        for feature in self.numeric_features:
            if feature in X_train.columns:
                if not pd.api.types.is_numeric_dtype(X_train[feature]):
                    X_train[feature] = pd.to_numeric(X_train[feature], errors='coerce')
                    X_test[feature] = pd.to_numeric(X_test[feature], errors='coerce')

                if X_train[feature].isnull().any():
                    if STANDARDIZATION_CONFIG['numeric_fill_strategy'] == 'median':
                        fill_val = X_train[feature].median()
                    elif STANDARDIZATION_CONFIG['numeric_fill_strategy'] == 'zero':
                        fill_val = 0
                    else:
                        fill_val = X_train[feature].mean()

                    if pd.isna(fill_val):
                        fill_val = 0
                    X_train[feature] = X_train[feature].fillna(fill_val)
                    X_test[feature] = X_test[feature].fillna(fill_val)

        # 处理分类特征
        for feature in self.string_features:
            if feature in X_train.columns:
                if not pd.api.types.is_string_dtype(X_train[feature]):
                    X_train[feature] = X_train[feature].astype(str)
                    X_test[feature] = X_test[feature].astype(str)

                if X_train[feature].isnull().any():
                    if STANDARDIZATION_CONFIG['categorical_fill_strategy'] == 'unknown':
                        fill_val = 'unknown'
                    else:
                        fill_val = X_train[feature].mode()[0]
                    X_train[feature] = X_train[feature].fillna(fill_val)
                    X_test[feature] = X_test[feature].fillna(fill_val)

        # 构建预处理管道
        transformers = []

        if self.numeric_features:
            transformers.append(('num', StandardScaler(
                with_mean=STANDARDIZATION_CONFIG['with_mean'],
                with_std=STANDARDIZATION_CONFIG['with_std']
            ), self.numeric_features))

        if self.string_features:
            all_categories = []
            for col in self.string_features:
                if STANDARDIZATION_CONFIG['use_all_categories']:
                    unique_vals = pd.concat([X_train[col], X_test[col]]).unique()
                else:
                    unique_vals = X_train[col].unique()
                unique_vals = [str(v) if pd.notna(v) else 'nan' for v in unique_vals]
                all_categories.append(sorted(set(unique_vals)))

            transformers.append((
                'cat',
                OneHotEncoder(
                    categories=all_categories,
                    handle_unknown=STANDARDIZATION_CONFIG['handle_unknown'],
                    sparse_output=STANDARDIZATION_CONFIG['sparse_output']
                ),
                self.string_features
            ))

        self.preprocessor = ColumnTransformer(transformers=transformers)

        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # 获取特征名称
        self.feature_names_processed = []
        if self.numeric_features:
            self.feature_names_processed.extend(self.numeric_features)

        if self.string_features and hasattr(self.preprocessor, 'named_transformers_'):
            cat_encoder = self.preprocessor.named_transformers_.get('cat')
            if cat_encoder and hasattr(cat_encoder, 'get_feature_names_out'):
                cat_features = cat_encoder.get_feature_names_out(self.string_features)
                self.feature_names_processed.extend(cat_features)

        self.X_train_processed = X_train_processed
        self.X_test_processed = X_test_processed

        return X_train_processed, X_test_processed

    def standardize_target(self):
        """标准化目标变量"""
        print("\n目标变量标准化...")

        y_train = self.train_raw[self.target_col].copy()
        y_test = self.test_raw[self.target_col].copy()

        if not pd.api.types.is_numeric_dtype(y_train):
            y_train = pd.to_numeric(y_train, errors='coerce')
            y_test = pd.to_numeric(y_test, errors='coerce')

        if y_train.isnull().any():
            if STANDARDIZATION_CONFIG['numeric_fill_strategy'] == 'median':
                fill_val = y_train.median()
            elif STANDARDIZATION_CONFIG['numeric_fill_strategy'] == 'zero':
                fill_val = 0
            else:
                fill_val = y_train.mean()

            if pd.isna(fill_val):
                fill_val = 0
            y_train = y_train.fillna(fill_val)
            y_test = y_test.fillna(fill_val)

        y_train_std = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_test_std = self.target_scaler.transform(y_test.values.reshape(-1, 1))

        self.y_train_processed = y_train_std.flatten()
        self.y_test_processed = y_test_std.flatten()

        return y_train_std, y_test_std

    def reconstruct_datasets(self):
        """重构数据集"""
        print("\n重构数据集...")

        self.train_processed = pd.DataFrame({self.name_col: self.train_raw[self.name_col].values})
        features_train_df = pd.DataFrame(self.X_train_processed, columns=self.feature_names_processed)
        self.train_processed = pd.concat([self.train_processed, features_train_df], axis=1)
        self.train_processed[self.target_col] = self.y_train_processed

        self.test_processed = pd.DataFrame({self.name_col: self.test_raw[self.name_col].values})
        features_test_df = pd.DataFrame(self.X_test_processed, columns=self.feature_names_processed)
        self.test_processed = pd.concat([self.test_processed, features_test_df], axis=1)
        self.test_processed[self.target_col] = self.y_test_processed

        return self.train_processed, self.test_processed

    def save_results(self):
        """保存结果"""
        print("\n保存标准化结果...")

        os.makedirs(self.output_dir, exist_ok=True)

        if OUTPUT_CONFIG['save_standardized_output']:
            self.train_processed.to_csv(self.paths['train_standardized_path'], index=False,
                                        encoding=STANDARDIZATION_CONFIG['csv_encoding'])
            print(f"✓ 标准化训练集已保存: {self.paths['train_standardized_path']}")

            self.test_processed.to_csv(self.paths['test_standardized_path'], index=False,
                                       encoding=STANDARDIZATION_CONFIG['csv_encoding'])
            print(f"✓ 标准化测试集已保存: {self.paths['test_standardized_path']}")

        if OUTPUT_CONFIG['save_log']:
            self.save_log()

    def save_log(self):
        """保存日志"""
        log_path = os.path.join(self.output_dir, 'processing_log.txt')

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("数据生成与标准化处理日志\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"输入文件夹: {self.paths['input_folder']}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")

            f.write("处理流程:\n")
            f.write("  1. VAE数据生成\n")
            f.write("  2. 数据标准化\n\n")

            f.write("VAE生成:\n")
            f.write(f"  输入文件: {self.paths['train_original_path']}\n")
            f.write(f"  生成样本数: {GENERATION_PARAMS['n_samples']}\n")
            f.write(f"  输出文件: {self.paths['vae_output_path']}\n\n")

            f.write("标准化:\n")
            f.write(f"  训练集: {self.paths['vae_output_path']}\n")
            f.write(f"  测试集: {self.paths['test_path']}\n")
            f.write(f"  输出目录: {self.output_dir}\n\n")

            f.write("标准化方式:\n")
            f.write("  数值特征: StandardScaler (mean=0, std=1)\n")
            f.write("  分类特征: OneHotEncoder (独热编码)\n")
            f.write("  目标变量: StandardScaler (mean=0, std=1)\n")

        print(f"✓ 日志已保存: {log_path}")

    def run(self):
        """运行标准化流程"""
        print("\n" + "=" * 60)
        print("开始数据标准化处理")
        print("=" * 60)

        try:
            self.load_data()
            self.identify_columns()
            self.preprocess_features()
            self.standardize_target()
            self.reconstruct_datasets()
            self.save_results()

            print("\n标准化处理完成！")
            return True
        except Exception as e:
            print(f"\n标准化处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False


class CustomVAE(nn.Module):
    """自定义VAE模型"""

    def __init__(self, input_dim, hidden_dim=256, latent_dim=64, dropout_rate=0.3):
        super(CustomVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
        )

        self.fc_mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 4, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE损失函数"""
    recon_loss = nn.MSELoss(reduction='mean')(recon_x, x)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div, recon_loss.item(), kl_div.item()


def main():
    """主函数"""
    print("=" * 80)
    print("VAE数据生成与标准化系统")
    print("=" * 80)

    # 获取自动路径
    paths = get_auto_paths()

    print(f"\n配置信息:")
    print(f"  输入文件夹: {paths['input_folder']}")
    print(f"  训练数据: {paths['train_original_path']}")
    print(f"  测试数据: {paths['test_path']}")
    print(f"  输出目录: {paths['output_dir']}")

    # 检查输入文件
    if not os.path.exists(paths['train_original_path']):
        print(f"\n错误: 未找到训练集文件 {paths['train_original_path']}")
        return

    if not os.path.exists(paths['test_path']):
        print(f"\n错误: 未找到测试集文件 {paths['test_path']}")
        return

    # 创建输出目录
    os.makedirs(paths['output_dir'], exist_ok=True)

    # 步骤1: VAE数据生成
    print("\n" + "=" * 80)
    print("步骤 1/2: VAE数据生成")
    print("=" * 80)

    vae_processor = VAEProcessor(paths)
    vae_result = vae_processor.run()

    if vae_result is None:
        print("VAE数据生成失败，流程终止")
        return

    # 步骤2: 数据标准化
    print("\n" + "=" * 80)
    print("步骤 2/2: 数据标准化")
    print("=" * 80)

    standardizer = DataStandardizer(paths)
    standardizer.run()

    # 显示最终结果
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)

    print(f"\n最终保存的文件:")
    print(f"  1. VAE生成数据: {paths['vae_output_path']}")
    print(f"  2. 标准化训练集: {paths['train_standardized_path']}")
    print(f"  3. 标准化测试集: {paths['test_standardized_path']}")

    if OUTPUT_CONFIG['save_log']:
        print(f"  4. 处理日志: {os.path.join(paths['output_dir'], 'processing_log.txt')}")

    print(f"\n所有文件保存在: {os.path.abspath(paths['output_dir'])}")
    print("=" * 80)


if __name__ == "__main__":
    # 检查依赖
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        print("安装必要的依赖包...")
        import subprocess

        subprocess.check_call(["pip", "install", "scipy", "scikit-learn"])
        print("依赖包安装完成!")

    # 运行主函数
    main()