import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import json
import joblib
import warnings

warnings.filterwarnings('ignore')


class QSARPFASProcessor:

    def __init__(self, file_path):
        self.load_data(file_path)
        print("=" * 60)
        print("=" * 60)
        print(f"  original data: {self.df.shape}")
        print(f"  columns: {list(self.df.columns)}")

    def load_data(self, file_path):

        # Data loading
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        else:
            self.df = pd.read_excel(file_path)

        self.original_df = self.df.copy()

        # save name column（the first column）
        self.name_column_name = self.df.columns[0]
        self.name_column = self.df.iloc[:, 0].copy()

        print(f"\n  finished data loading:")
        print(f"  original data: {self.df.shape}")
        print(f"  Name column: '{self.name_column_name}' ({len(self.name_column)} values)")

    def identify_features_target(self):

        # target column
        target_cols = ['Kobs']
        self.target_col = None
        for col in target_cols:
            if col in self.df.columns:
                self.target_col = col
                break

        if self.target_col is None:
            self.target_col = self.df.columns[-1]
            print(
                f"Warning: The 'Kobs' column was not found; use the last column '{self.target_col}' as the target column.")

        # Name column
        self.name_col = self.name_column_name

        # Features columns
        self.feature_cols = [col for col in self.df.columns if col not in [self.name_col, self.target_col]]

        # Separating numeric features and string features
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

        print(f"\nfeature analysis:")
        print(f"  Name column: {self.name_col}")
        print(f"  target column: {self.target_col}")
        print(f"  features columns number: {len(self.feature_cols)}")
        print(
            f"  numeric features ({len(self.numeric_features)}): {self.numeric_features[:5]}{'...' if len(self.numeric_features) > 5 else ''}")
        print(
            f"  string features ({len(self.string_features)}): {self.string_features[:5]}{'...' if len(self.string_features) > 5 else ''}")

        return self.feature_cols, self.target_col, self.name_col

    # Splitting training data and testing data
    def split_data(self, train_ratio=0.8, test_ratio=0.2, random_state=42):

        X = self.df[self.feature_cols].copy()
        y = self.df[self.target_col].copy()
        names = self.df[self.name_col].copy()

        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Error: Target variable '{self.target_col}' is not numeric type")

        print(f"random_state: {random_state}")

        total_samples = len(X)

        # 向下取整计算测试集数量，训练集 = 总数 - 测试集
        test_count = int(np.floor(total_samples * test_ratio))
        train_count = total_samples - test_count

        print(f"Calculated split: test_count={test_count}, train_count={train_count}")

        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            X, y, names,
            test_size=test_count,
            random_state=random_state
        )

        self.names_train = names_train.reset_index(drop=True)
        self.names_test = names_test.reset_index(drop=True)

        print(f"total data: {total_samples}")
        print(f"training data: {train_count}  ({train_count / total_samples:.1%})")
        print(f"testing data: {test_count}  ({test_count / total_samples:.1%})")
        print(f"\ntarget split ratio: training={train_ratio:.0%}, testing={test_ratio:.0%}")
        print(
            f"actual split ratio: training={train_count / total_samples:.1%}, testing={test_count / total_samples:.1%}")

        # save
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

        print(f"\noriginal data:")
        print(f"  training: {self.train_raw.shape}")
        print(f"  testing: {self.test_raw.shape}")

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

        # 1. Handle numeric features
        if self.numeric_features:
            print(f"\nNumeric feature imputation:")
            for feature in self.numeric_features:
                if feature in X_train_filled.columns:
                    # Calculate training set mean
                    train_mean = X_train_filled[feature].mean()
                    self.train_stats['numeric_means'][feature] = train_mean

                    # Count missing values
                    train_missing = X_train_filled[feature].isnull().sum()
                    test_missing = X_test_filled[feature].isnull().sum()

                    # Fill missing values
                    if train_missing > 0:
                        X_train_filled[feature] = X_train_filled[feature].fillna(train_mean)
                        print(
                            f"  {feature}: Filled {train_missing} missing values in training set (mean: {train_mean:.4f})")
                        total_filled += train_missing

                    if test_missing > 0:
                        X_test_filled[feature] = X_test_filled[feature].fillna(train_mean)
                        print(f"  {feature}: Filled {test_missing} missing values in test set")
                        total_filled += test_missing

        # 2. Handle string features
        if self.string_features:
            print(f"\nString feature imputation:")
            for feature in self.string_features:
                if feature in X_train_filled.columns:
                    # Calculate training set mode
                    train_mode = X_train_filled[feature].mode()
                    if not train_mode.empty:
                        mode_value = train_mode[0]
                        self.train_stats['string_modes'][feature] = mode_value

                        # Count missing values
                        train_missing = X_train_filled[feature].isnull().sum()
                        test_missing = X_test_filled[feature].isnull().sum()

                        # Fill missing values
                        if train_missing > 0:
                            X_train_filled[feature] = X_train_filled[feature].fillna(mode_value)
                            print(
                                f"  {feature}: Filled {train_missing} missing values in training set (mode: '{mode_value}')")
                            total_filled += train_missing

                        if test_missing > 0:
                            X_test_filled[feature] = X_test_filled[feature].fillna(mode_value)
                            print(f"  {feature}: Filled {test_missing} missing values in test set")
                            total_filled += test_missing
                    else:
                        print(f"  {feature}: Warning - Unable to calculate mode, possibly too many missing values")

        # 3. Handle target variable
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
        """
        Feature preprocessing:
        Numeric features: Standard Scaling (mean=0, variance=1)
        String features: One-Hot Encoding
        """
        print("\n" + "=" * 60)
        print("Feature Preprocessing (Standardization)")
        print("=" * 60)
        print("Numeric features: StandardScaler (mean=0, std=1)")
        print("String features: OneHotEncoder")

        # Create preprocessing pipeline
        transformers = []

        if self.numeric_features:
            # Apply StandardScaler
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

        # Fit preprocessor on training set and transform both sets
        X_train_processed = self.preprocessor.fit_transform(self.X_train_filled)
        X_test_processed = self.preprocessor.transform(self.X_test_filled)

        # Get feature names
        self.feature_names_processed = []

        # Get numeric feature names
        if self.numeric_features:
            self.feature_names_processed.extend(self.numeric_features)

        # Get one-hot encoded feature names
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

        # Target variable standardization
        self.scaler_y = StandardScaler()
        y_train_processed = self.scaler_y.fit_transform(self.y_train_filled.values.reshape(-1, 1))
        y_test_processed = self.scaler_y.transform(self.y_test_filled.values.reshape(-1, 1))

        print(f"\nTarget variable standardization completed:")
        print(f"  Training set mean: {self.scaler_y.mean_[0]:.4f}")
        print(f"  Training set std: {np.sqrt(self.scaler_y.var_[0]):.4f}")
        print(f"  Standardized range: [{y_train_processed.min():.4f}, {y_train_processed.max():.4f}]")

        # Save results
        self.X_train_processed = X_train_processed
        self.X_test_processed = X_test_processed

        self.y_train_processed = y_train_processed
        self.y_test_processed = y_test_processed

        # Save scalers to output directory
        self._save_scalers()

        return (X_train_processed, X_test_processed, y_train_processed, y_test_processed)

    def _save_scalers(self):

        scaler_dir = os.path.join(self.output_dir, 'scalers')
        os.makedirs(scaler_dir, exist_ok=True)

        # 1. Save feature preprocessor (ColumnTransformer)
        if hasattr(self, 'preprocessor'):
            joblib.dump(self.preprocessor, os.path.join(scaler_dir, 'feature_preprocessor.pkl'))
            print(f"\nFeature preprocessor saved: {scaler_dir}/feature_preprocessor.pkl")

        # 2. Save target variable scaler
        if hasattr(self, 'scaler_y'):
            joblib.dump(self.scaler_y, os.path.join(scaler_dir, 'target_scaler.pkl'))
            print(f"Target scaler saved: {scaler_dir}/target_scaler.pkl")

        # 3. Save feature names list
        if hasattr(self, 'feature_names_processed') and self.feature_names_processed:
            feature_names_df = pd.DataFrame({
                'feature_index': range(len(self.feature_names_processed)),
                'feature_name': self.feature_names_processed
            })
            feature_names_df.to_csv(
                os.path.join(scaler_dir, 'feature_names.csv'),
                index=False,
                encoding='utf-8-sig'
            )
            print(f"Feature names list saved: {scaler_dir}/feature_names.csv")

        # 4. Save numeric features list
        if hasattr(self, 'numeric_features') and self.numeric_features:
            numeric_features_df = pd.DataFrame({
                'feature_name': self.numeric_features,
                'feature_type': 'numeric'
            })
            numeric_features_df.to_csv(
                os.path.join(scaler_dir, 'numeric_features.csv'),
                index=False,
                encoding='utf-8-sig'
            )
            print(f"Numeric features list saved: {scaler_dir}/numeric_features.csv")

        # 5. Save string features list
        if hasattr(self, 'string_features') and self.string_features:
            string_features_df = pd.DataFrame({
                'feature_name': self.string_features,
                'feature_type': 'categorical'
            })
            string_features_df.to_csv(
                os.path.join(scaler_dir, 'string_features.csv'),
                index=False,
                encoding='utf-8-sig'
            )
            print(f"String features list saved: {scaler_dir}/string_features.csv")

        # 6. Save scaler metadata
        scaler_meta = {
            'creation_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'numeric_features_count': len(self.numeric_features) if hasattr(self, 'numeric_features') else 0,
            'string_features_count': len(self.string_features) if hasattr(self, 'string_features') else 0,
            'total_processed_features': len(self.feature_names_processed) if hasattr(self,
                                                                                     'feature_names_processed') else 0,
            'target_column': self.target_col if hasattr(self, 'target_col') else None
        }

        meta_path = os.path.join(scaler_dir, 'scaler_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(scaler_meta, f, ensure_ascii=False, indent=4)
        print(f"Scaler metadata saved: {scaler_dir}/scaler_metadata.json")

        print(f"\nAll scalers saved to: {os.path.abspath(scaler_dir)}/")

    def reconstruct_datasets(self):

        print("\n" + "=" * 60)
        print("Reconstructing Datasets with Name Column")
        print("=" * 60)

        # Verify data correspondence
        print(f"Verifying data correspondence:")
        print(f"  Training set Name count: {len(self.names_train)}")
        print(f"  Training set features count: {len(self.X_train_processed)}")
        print(f"  Training set target count: {len(self.y_train_processed)}")

        # Reconstruct processed training dataset
        self.train_processed = pd.DataFrame({self.name_col: self.names_train})

        # Add processed features
        processed_features_train = pd.DataFrame(
            self.X_train_processed,
            columns=self.feature_names_processed
        )
        self.train_processed = pd.concat([self.train_processed, processed_features_train], axis=1)

        # Add target variable
        self.train_processed[self.target_col] = self.y_train_processed.flatten()

        # Reconstruct processed test dataset
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

        # Display first few rows to verify correspondence
        print(f"\nTraining set first 3 rows - Name verification:")
        for i in range(min(3, len(self.train_processed))):
            print(
                f"  Row {i + 1}: Name='{self.train_processed.iloc[i][self.name_col]}', {self.target_col}={self.train_processed.iloc[i][self.target_col]:.4f}")

        return self.train_processed, self.test_processed

    def save_datasets(self, output_dir='./QSAR_PFAS_processed/new'):

        print("\n" + "=" * 60)
        print("Saving Processed Results")
        print("=" * 60)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save raw datasets
        self.train_raw.to_csv(f'{output_dir}/train_raw.csv', index=False, encoding='utf-8-sig')
        self.test_raw.to_csv(f'{output_dir}/test_raw.csv', index=False, encoding='utf-8-sig')

        print(f"Raw datasets saved:")
        print(f"  {output_dir}/train_raw.csv - Raw training set ({self.train_raw.shape})")
        print(f"  {output_dir}/test_raw.csv - Raw test set ({self.test_raw.shape})")

        # 2. Save standardized datasets
        self.train_processed.to_csv(f'{output_dir}/train_standardized.csv', index=False, encoding='utf-8-sig')
        self.test_processed.to_csv(f'{output_dir}/test_standardized.csv', index=False, encoding='utf-8-sig')

        print(f"\nStandardized datasets saved:")
        print(f"  {output_dir}/train_standardized.csv - Standardized training set ({self.train_processed.shape})")
        print(f"  {output_dir}/test_standardized.csv - Standardized test set ({self.test_processed.shape})")

        # 3. Save standardization parameters
        if hasattr(self, 'scaler_y'):
            scaler_info = pd.DataFrame({
                'target_column': [self.target_col],
                'original_mean': [self.scaler_y.mean_[0]],
                'original_std': [np.sqrt(self.scaler_y.var_[0])],
                'standardized_mean': [0],
                'standardized_std': [1]
            })
            scaler_info.to_csv(f'{output_dir}/standardization_params.csv', index=False, encoding='utf-8-sig')
            print(f"\nStandardization parameters saved: {output_dir}/standardization_params.csv")

        # 4. Save data correspondence verification file
        verification_data = []
        for i in range(min(10, len(self.train_raw))):
            verification_data.append({
                'index': i,
                'raw_name': self.train_raw.iloc[i][self.name_col],
                'processed_name': self.train_processed.iloc[i][self.name_col],
                'raw_target': self.train_raw.iloc[i][self.target_col],
                'processed_target': self.train_processed.iloc[i][self.target_col],
                'match': self.train_raw.iloc[i][self.name_col] == self.train_processed.iloc[i][self.name_col]
            })

        verification_df = pd.DataFrame(verification_data)
        verification_df.to_csv(f'{output_dir}/data_correspondence_verification.csv', index=False, encoding='utf-8-sig')
        print(f"Data correspondence verification file saved: {output_dir}/data_correspondence_verification.csv")

        # 5. Save feature standardization parameters
        if hasattr(self, 'preprocessor') and 'num' in self.preprocessor.named_transformers_:
            num_scaler = self.preprocessor.named_transformers_['num']
            if hasattr(num_scaler, 'mean_') and hasattr(num_scaler, 'scale_'):
                feature_scaler_info = pd.DataFrame({
                    'feature': self.numeric_features,
                    'original_mean': num_scaler.mean_,
                    'original_std': num_scaler.scale_,
                    'standardized_mean': 0,
                    'standardized_std': 1
                })
                feature_scaler_info.to_csv(f'{output_dir}/feature_standardization_params.csv',
                                           index=False, encoding='utf-8-sig')
                print(f"Feature standardization parameters saved: {output_dir}/feature_standardization_params.csv")

        # 6. Save processing log
        log_filepath = os.path.join(output_dir, 'processing_log.txt')
        with open(log_filepath, 'w', encoding='utf-8') as f:
            f.write("QSAR-PFAS Data Processing Log\n")
            f.write("=" * 50 + "\n\n")

            f.write("Processing Configuration:\n")
            f.write("  - Data split: Training 80% : Test 20%\n")
            f.write("  - Feature processing: Numeric features standardized, String features one-hot encoded\n")
            f.write("  - Target variable: Standardized\n\n")

            # Basic information
            f.write("Basic Information:\n")
            f.write(f"  Processing time: {pd.Timestamp.now()}\n")
            f.write(f"  Original data shape: {self.original_df.shape}\n")
            f.write(f"  Name column: {self.name_col}\n")
            f.write(f"  Target column: {self.target_col}\n")
            f.write(f"  Number of numeric features: {len(self.numeric_features)}\n")
            f.write(f"  Number of string features: {len(self.string_features)}\n\n")

            # Dataset information
            f.write("Dataset Information:\n")
            f.write(f"  Training set: {len(self.train_processed)} samples\n")
            f.write(f"  Test set: {len(self.test_processed)} samples\n")
            f.write(f"  Total samples: {len(self.train_processed) + len(self.test_processed)}\n\n")

            # Standardization information
            if hasattr(self, 'scaler_y'):
                f.write("Target Variable Standardization Information:\n")
                f.write(f"  Original mean: {self.scaler_y.mean_[0]:.4f}\n")
                f.write(f"  Original std: {np.sqrt(self.scaler_y.var_[0]):.4f}\n")
                f.write(f"  After standardization: mean=0, std=1\n\n")

            # Data correspondence verification
            f.write("Data Correspondence Verification Results:\n")
            matches = verification_df['match'].sum()
            total = len(verification_df)
            f.write(f"  Verified {total} records, {matches} matched, {total - matches} mismatched\n")
            if matches == total:
                f.write("  [PASS] All Name columns correctly correspond to data\n")
            else:
                f.write("  [WARNING] Name column mismatch detected\n")

            # Processing steps
            f.write("\nProcessing Steps:\n")
            f.write("  1. Load data, identify Name column\n")
            f.write("  2. Split X, y and names simultaneously to ensure correspondence (80:20)\n")
            f.write("  3. Fill missing values (numeric: mean, string: mode)\n")
            f.write("  4. Feature preprocessing (numeric: standardization, string: one-hot encoding)\n")
            f.write("  5. Target variable standardization\n")
            f.write("  6. Save scalers as pkl files in output directory\n")
            f.write("  7. Reconstruct datasets with Name column\n")
            f.write("  8. Save processed results\n\n")

        print(f"Processing log saved: {log_filepath}")
        print(f"\nAll files saved to: {os.path.abspath(output_dir)}/")

        return output_dir

    def run_pipeline(self, random_state=42):

        print(f"\n{'=' * 60}")
        print("Starting QSAR-PFAS Data Processing Pipeline")
        print(f"{'=' * 60}")
        print("Configuration:")
        print("  - Data split: Training 80% : Test 20%")
        print("  - Feature processing: Standardization (StandardScaler)")
        print("  - Target variable: Standardization")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - Scaler directory: {self.output_dir}/scalers/")
        print(f"{'=' * 60}")

        try:
            # 1. Identify features and target
            print("\n1. Identifying features and target")
            self.identify_features_target()

            # 2. Split data
            print("\n2. Splitting data (80:20)")
            self.split_data(train_ratio=0.8, test_ratio=0.2, random_state=random_state)

            # 3. Fill missing values
            print("\n3. Filling missing values...")
            self.fill_missing_values()

            # 4. Feature preprocessing
            print("\n4. Preprocessing features")
            self.preprocess_features()

            # 5. Reconstruct datasets
            print("\n5. Reconstructing datasets")
            self.reconstruct_datasets()

            # 6. Save results
            print("\n6. Saving processed results...")
            output_dir = self.save_datasets(output_dir=self.output_dir)

            print(f"\n{'=' * 60}")
            print("Data Processing Completed!")
            print(f"{'=' * 60}")

            print(f"\nData Processing Summary:")
            print(f"  - Original data: {self.original_df.shape[0]} rows x {self.original_df.shape[1]} columns")
            print(f"  - Name column: {self.name_col}")
            print(f"  - Training set: {self.train_processed.shape}")
            print(f"  - Test set: {self.test_processed.shape}")
            print(f"  - Processing method: Numeric features standardized, string features one-hot encoded")
            print(f"  - Standardization: mean=0, std=1")
            print(f"  - Output directory: {os.path.abspath(output_dir)}/")
            print(f"  - Scaler directory: {os.path.abspath(output_dir)}/scalers/")

            return output_dir

        except Exception as e:
            print(f"\nError occurred during processing: {e}")
            import traceback
            traceback.print_exc()
            return None


# Usage example
if __name__ == "__main__":
    file_path = "Original_change.csv"
    processor = QSARPFASProcessor(file_path)

    processor.output_dir = './pre/renew/1930'

    output_dir = processor.run_pipeline(random_state=1930)

    if output_dir:
        print(f"\nAll results:")
        print(f"  1. train_raw.csv")
        print(f"  2. test_raw.csv")
        print(f"  3. train_standardized.csv")
        print(f"  4. test_standardized.csv")
        print(f"  5. standardization_params.csv")
        print(f"  6. feature_standardization_params.csv")
        print(f"  7. data_correspondence_verification.csv")
        print(f"  8. processing_log.txt")
        print(f"\nScaler files (in {output_dir}/scalers/):")
        print(f"  9. feature_preprocessor.pkl")
        print(f"  10. target_scaler.pkl")
        print(f"  11. feature_names.csv")
        print(f"  12. numeric_features.csv")
        print(f"  13. string_features.csv")
        print(f"  14. scaler_metadata.json")
