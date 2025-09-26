class DataOperations:
    def preprocess(self, df, target_col=None, cardinality_thresh=50, na_thresh=0.5):
        """
        Robust preprocessing:
        - Drop columns with high cardinality (likely IDs/names)
        - Drop columns with >na_thresh missing values
        - Fill missing values (median for numeric, mode for categorical)
        - Encode all object/categorical columns except target
        - Drop any columns that are still not numeric
        """
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        # 1. Drop high cardinality columns
        high_card_cols = [col for col in df.columns if df[col].nunique() > cardinality_thresh and col != target_col]
        df = df.drop(columns=high_card_cols)
        # 2. Drop columns with too many missing values
        na_limit = len(df) * na_thresh
        high_na_cols = [col for col in df.columns if col != target_col and df[col].isna().sum() > na_limit]
        df = df.drop(columns=high_na_cols)
        # 3. Fill missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        # 4. Encode all object/categorical columns except target
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col != target_col:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        # 5. Drop any columns that are still not numeric
        non_numeric_cols = [col for col in df.columns if col != target_col and not pd.api.types.is_numeric_dtype(df[col])]
        df = df.drop(columns=non_numeric_cols)
        return df
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
    from imblearn.over_sampling import SMOTE

    def drop_columns(self, df, columns):
        """Drop specified columns from DataFrame."""
        return df.drop(columns=columns)

    def scale(self, df, columns=None, method='standard'):
        """Scale specified columns using Standard or MinMax scaler."""
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        cols = columns if columns else df.select_dtypes(include='number').columns
        df[cols] = scaler.fit_transform(df[cols])
        return df

    def encode(self, df, columns=None):
        """One-hot encode specified categorical columns."""
        cols = columns if columns else df.select_dtypes(include='object').columns
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cols), index=df.index)
        df = df.drop(columns=cols)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    def oversample(self, X, y):
        """Apply SMOTE oversampling to balance classes."""
        smote = SMOTE()
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res

    def generate_features(self, df):
        """Basic feature generation: add interaction terms for numeric columns."""
        num_cols = df.select_dtypes(include='number').columns
        for i, col1 in enumerate(num_cols):
            for col2 in num_cols[i+1:]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        return df
