import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import warnings

warnings.filterwarnings('ignore')

FILE_PATH = "../Data/PFAS.csv"
OUTPUT_DIR = '../Data/Splits/859'
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_STATE = 859
TARGET_COLUMN = "Kobs"


class QSARPFASProcessor:

    def __init__(self, file_path, output_dir, random_state):
        self.output_dir = output_dir
        self.random_state = random_state
        self.load_data(file_path)
        print("=" * 60)
        print(f"  Original data: {self.df.shape}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Random state: {self.random_state}")

    def load_data(self, file_path):
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        else:
            self.df = pd.read_excel(file_path)

        self.original_df = self.df.copy()
        self.name_column_name = self.df.columns[0]
        self.name_column = self.df.iloc[:, 0].copy()

        print(f"\nData loading completed:")
        print(f"  Original data: {self.df.shape}")
        print(f"  Name column: '{self.name_column_name}' ({len(self.name_column)} values)")

    def identify_features_target(self):
        target_cols = [TARGET_COLUMN]
        self.target_col = None
        for col in target_cols:
            if col in self.df.columns:
                self.target_col = col
                break

        if self.target_col is None:
            self.target_col = self.df.columns[-1]
            print(f"Warning: '{TARGET_COLUMN}' column not found, using last column '{self.target_col}' as target.")

        self.name_col = self.name_column_name
        self.feature_cols = [col for col in self.df.columns if col not in [self.name_col, self.target_col]]

        self.numeric_features = []
        self.string_features = []

        for col in self.feature_cols:
            col_data = self.df[col]
            try:
                numeric_test = pd.to_numeric(col_data, errors='coerce')
                non_na_ratio = numeric_test.notna().mean()
                if non_na_ratio > 0.9:
                    self.numeric_features.append(col)
                    self.df[col] = numeric_test
                else:
                    self.string_features.append(col)
            except:
                self.string_features.append(col)

        print(f"\nFeature analysis:")
        print(f"  Name column: {self.name_col}")
        print(f"  Target column: {self.target_col}")
        print(f"  Feature columns: {len(self.feature_cols)}")
        print(f"  Numeric features: {len(self.numeric_features)}")
        print(f"  String features: {len(self.string_features)}")

        return self.feature_cols, self.target_col, self.name_col

    def split_data(self):
        X = self.df[self.feature_cols].copy()
        y = self.df[self.target_col].copy()
        names = self.df[self.name_col].copy()

        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Error: Target variable '{self.target_col}' is not numeric type")

        total_samples = len(X)
        test_count = int(np.floor(total_samples * TEST_RATIO))
        train_count = total_samples - test_count

        print(f"\nData split:")
        print(f"  Random state: {self.random_state}")
        print(f"  Total samples: {total_samples}")
        print(f"  Test count: {test_count}")
        print(f"  Train count: {train_count}")

        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            X, y, names,
            test_size=test_count,
            random_state=self.random_state
        )

        self.names_train = names_train.reset_index(drop=True)
        self.names_test = names_test.reset_index(drop=True)

        print(f"  Training: {train_count} ({train_count / total_samples:.1%})")
        print(f"  Testing: {test_count} ({test_count / total_samples:.1%})")

        self.X_train_raw = X_train.reset_index(drop=True).copy()
        self.X_test_raw = X_test.reset_index(drop=True).copy()
        self.y_train_raw = y_train.reset_index(drop=True).copy()
        self.y_test_raw = y_test.reset_index(drop=True).copy()

        self.train_raw = pd.concat([
            pd.DataFrame({self.name_col: self.names_train}),
            self.X_train_raw,
            pd.DataFrame({self.target_col: self.y_train_raw})
        ], axis=1)

        self.test_raw = pd.concat([
            pd.DataFrame({self.name_col: self.names_test}),
            self.X_test_raw,
            pd.DataFrame({self.target_col: self.y_test_raw})
        ], axis=1)

        print(f"\nRaw datasets:")
        print(f"  Training: {self.train_raw.shape}")
        print(f"  Testing: {self.test_raw.shape}")

        return (X_train, X_test, y_train, y_test)

    def fill_missing_values(self):
        print("\n" + "=" * 60)
        print("Missing Value Imputation")
        print("=" * 60)

        X_train_filled = self.X_train_raw.copy()
        X_test_filled = self.X_test_raw.copy()

        self.train_stats = {
            'numeric_means': {},
            'string_modes': {}
        }

        total_filled = 0

        if self.numeric_features:
            print(f"\nNumeric feature imputation:")
            for feature in self.numeric_features:
                if feature in X_train_filled.columns:
                    train_mean = X_train_filled[feature].mean()
                    self.train_stats['numeric_means'][feature] = train_mean

                    train_missing = X_train_filled[feature].isnull().sum()
                    test_missing = X_test_filled[feature].isnull().sum()

                    if train_missing > 0:
                        X_train_filled[feature] = X_train_filled[feature].fillna(train_mean)
                        print(f"  {feature}: Filled {train_missing} missing values in training set (mean: {train_mean:.4f})")
                        total_filled += train_missing

                    if test_missing > 0:
                        X_test_filled[feature] = X_test_filled[feature].fillna(train_mean)
                        print(f"  {feature}: Filled {test_missing} missing values in test set")
                        total_filled += test_missing

        if self.string_features:
            print(f"\nString feature imputation:")
            for feature in self.string_features:
                if feature in X_train_filled.columns:
                    train_mode = X_train_filled[feature].mode()
                    if not train_mode.empty:
                        mode_value = train_mode[0]
                        self.train_stats['string_modes'][feature] = mode_value

                        train_missing = X_train_filled[feature].isnull().sum()
                        test_missing = X_test_filled[feature].isnull().sum()

                        if train_missing > 0:
                            X_train_filled[feature] = X_train_filled[feature].fillna(mode_value)
                            print(f"  {feature}: Filled {train_missing} missing values in training set (mode: '{mode_value}')")
                            total_filled += train_missing

                        if test_missing > 0:
                            X_test_filled[feature] = X_test_filled[feature].fillna(mode_value)
                            print(f"  {feature}: Filled {test_missing} missing values in test set")
                            total_filled += test_missing
                    else:
                        print(f"  {feature}: Warning - Unable to calculate mode")

        y_train_mean = self.y_train_raw.mean()
        y_train_filled = self.y_train_raw.fillna(y_train_mean)
        y_test_filled = self.y_test_raw.fillna(y_train_mean)

        target_filled = self.y_train_raw.isnull().sum() + self.y_test_raw.isnull().sum()
        if target_filled > 0:
            print(f"\nTarget variable '{self.target_col}' imputation: Using mean {y_train_mean:.4f}")
            print(f"  Total {target_filled} missing values filled")
            total_filled += target_filled

        print(f"\nTotal missing values filled: {total_filled}")

        self.X_train_filled = X_train_filled
        self.X_test_filled = X_test_filled
        self.y_train_filled = y_train_filled
        self.y_test_filled = y_test_filled

        return (X_train_filled, X_test_filled, y_train_filled, y_test_filled)

    def preprocess_features(self):
        print("\n" + "=" * 60)
        print("Feature Preprocessing")
        print("=" * 60)
        print("Numeric features: StandardScaler (mean=0, std=1)")
        print("String features: OneHotEncoder")

        transformers = []

        if self.numeric_features:
            transformers.append(('num', StandardScaler(), self.numeric_features))
            print(f"Numeric features: StandardScaler applied")

        if self.string_features:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.string_features))
            print(f"String features: OneHotEncoder applied")

        if transformers:
            self.preprocessor = ColumnTransformer(transformers=transformers)
        else:
            raise ValueError("No usable features found")

        X_train_processed = self.preprocessor.fit_transform(self.X_train_filled)
        X_test_processed = self.preprocessor.transform(self.X_test_filled)

        self.feature_names_processed = []

        if self.numeric_features:
            self.feature_names_processed.extend(self.numeric_features)

        if self.string_features and hasattr(self.preprocessor, 'named_transformers_'):
            if 'cat' in self.preprocessor.named_transformers_:
                cat_encoder = self.preprocessor.named_transformers_['cat']
                if hasattr(cat_encoder, 'get_feature_names_out'):
                    cat_features = cat_encoder.get_feature_names_out(self.string_features)
                    self.feature_names_processed.extend(cat_features)

        print(f"\nPreprocessing completed:")
        print(f"  Original features: {len(self.feature_cols)}")
        print(f"  Processed features: {len(self.feature_names_processed)}")
        print(f"  Training set shape: {X_train_processed.shape}")
        print(f"  Test set shape: {X_test_processed.shape}")

        self.scaler_y = StandardScaler()
        y_train_processed = self.scaler_y.fit_transform(self.y_train_filled.values.reshape(-1, 1))
        y_test_processed = self.scaler_y.transform(self.y_test_filled.values.reshape(-1, 1))

        print(f"\nTarget variable standardization completed:")
        print(f"  Training set mean: {self.scaler_y.mean_[0]:.4f}")
        print(f"  Training set std: {np.sqrt(self.scaler_y.var_[0]):.4f}")

        self.X_train_processed = X_train_processed
        self.X_test_processed = X_test_processed
        self.y_train_processed = y_train_processed
        self.y_test_processed = y_test_processed

        return (X_train_processed, X_test_processed, y_train_processed, y_test_processed)

    def reconstruct_datasets(self):
        print("\n" + "=" * 60)
        print("Reconstructing Datasets with Name Column")
        print("=" * 60)

        print(f"Verifying data correspondence:")
        print(f"  Training set Name count: {len(self.names_train)}")
        print(f"  Training set features count: {len(self.X_train_processed)}")
        print(f"  Training set target count: {len(self.y_train_processed)}")

        if len(self.names_train) == len(self.X_train_processed) == len(self.y_train_processed):
            print(f"  [OK] Training set: All data counts match")
        else:
            print(f"  [FAIL] Training set: Data count mismatch!")
            raise ValueError("Data correspondence verification failed!")

        if len(self.names_test) == len(self.X_test_processed) == len(self.y_test_processed):
            print(f"  [OK] Test set: All data counts match")
        else:
            print(f"  [FAIL] Test set: Data count mismatch!")
            raise ValueError("Data correspondence verification failed!")

        self.train_processed = pd.DataFrame({self.name_col: self.names_train})
        processed_features_train = pd.DataFrame(
            self.X_train_processed,
            columns=self.feature_names_processed
        )
        self.train_processed = pd.concat([self.train_processed, processed_features_train], axis=1)
        self.train_processed[self.target_col] = self.y_train_processed.flatten()

        self.test_processed = pd.DataFrame({self.name_col: self.names_test})
        processed_features_test = pd.DataFrame(
            self.X_test_processed,
            columns=self.feature_names_processed
        )
        self.test_processed = pd.concat([self.test_processed, processed_features_test], axis=1)
        self.test_processed[self.target_col] = self.y_test_processed.flatten()

        print(f"\nDataset reconstruction completed:")
        print(f"  Training set: {self.train_processed.shape}")
        print(f"  Test set: {self.test_processed.shape}")

        return self.train_processed, self.test_processed

    def save_datasets(self):
        print("\n" + "=" * 60)
        print("Saving Processed Results")
        print("=" * 60)

        os.makedirs(self.output_dir, exist_ok=True)

        train_raw_filename = f"train_{self.random_state}_raw.csv"
        test_raw_filename = f"test_{self.random_state}_raw.csv"
        train_std_filename = f"train_{self.random_state}_standardized.csv"
        test_std_filename = f"test_{self.random_state}_standardized.csv"

        self.train_raw.to_csv(f'{self.output_dir}/{train_raw_filename}', index=False, encoding='utf-8-sig')
        self.test_raw.to_csv(f'{self.output_dir}/{test_raw_filename}', index=False, encoding='utf-8-sig')

        print(f"Raw datasets saved:")
        print(f"  {self.output_dir}/{train_raw_filename} ({self.train_raw.shape})")
        print(f"  {self.output_dir}/{test_raw_filename} ({self.test_raw.shape})")

        self.train_processed.to_csv(f'{self.output_dir}/{train_std_filename}', index=False, encoding='utf-8-sig')
        self.test_processed.to_csv(f'{self.output_dir}/{test_std_filename}', index=False, encoding='utf-8-sig')

        print(f"\nStandardized datasets saved:")
        print(f"  {self.output_dir}/{train_std_filename} ({self.train_processed.shape})")
        print(f"  {self.output_dir}/{test_std_filename} ({self.test_processed.shape})")

        print(f"\nAll files saved to: {os.path.abspath(self.output_dir)}/")

        return self.output_dir

    def run_pipeline(self):
        print(f"\n{'=' * 60}")
        print("Starting QSAR-PFAS Data Processing Pipeline")
        print(f"{'=' * 60}")
        print(f"  Data file: {FILE_PATH}")
        print(f"  Output directory: {OUTPUT_DIR}")
        print(f"  Split ratio: {TRAIN_RATIO:.0%} : {TEST_RATIO:.0%}")
        print(f"  Random state: {self.random_state}")
        print(f"{'=' * 60}")

        try:
            print("\n[Step 1] Identifying features and target")
            self.identify_features_target()

            print("\n[Step 2] Splitting data")
            self.split_data()

            print("\n[Step 3] Filling missing values")
            self.fill_missing_values()

            print("\n[Step 4] Preprocessing features")
            self.preprocess_features()

            print("\n[Step 5] Reconstructing datasets")
            self.reconstruct_datasets()

            print("\n[Step 6] Saving processed results")
            output_dir = self.save_datasets()

            print(f"\n{'=' * 60}")
            print("Data Processing Completed!")
            print(f"{'=' * 60}")
            print(f"  Original data: {self.original_df.shape}")
            print(f"  Training set: {self.train_processed.shape}")
            print(f"  Test set: {self.test_processed.shape}")
            print(f"  Output directory: {os.path.abspath(output_dir)}/")

            return output_dir

        except Exception as e:
            print(f"\nError occurred during processing: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    processor = QSARPFASProcessor(FILE_PATH, OUTPUT_DIR, RANDOM_STATE)
    output_dir = processor.run_pipeline()

    if output_dir:
        print(f"\nAll results:")
        print(f"  1. train_{RANDOM_STATE}_raw.csv")
        print(f"  2. test_{RANDOM_STATE}_raw.csv")
        print(f"  3. train_{RANDOM_STATE}_standardized.csv")
        print(f"  4. test_{RANDOM_STATE}_standardized.csv")
