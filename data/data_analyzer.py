import pandas as pd
import numpy as np

class DataAnalyzer:
    def analyze(self, data, metrics=None):
        analysis = {}
        # Basic info
        analysis['head'] = data.head().to_dict()
        analysis['tail'] = data.tail().to_dict()
        analysis['description'] = data.describe(include='all').to_dict()
        analysis['shape'] = data.shape
        analysis['columns'] = list(data.columns)
        # Feature types
        analysis['feature_types'] = {
            'numerical': list(data.select_dtypes(include=np.number).columns),
            'categorical': list(data.select_dtypes(include='object').columns),
            'datetime': list(data.select_dtypes(include='datetime').columns)
        }
        # Missing values
        analysis['missing_values'] = data.isnull().sum().to_dict()
        analysis['missing_pct'] = (data.isnull().mean() * 100).round(2).to_dict()
        # Outlier detection (IQR method)
        outliers = {}
        for col in analysis['feature_types']['numerical']:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers[col] = int(((data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)).sum())
        analysis['outliers'] = outliers
        # Correlation
        # corr = data.corr()
        corr = data.select_dtypes(include='number').corr()
        analysis['correlation'] = corr.to_dict()
        # Top correlated pairs
        corr_pairs = corr.abs().unstack().sort_values(ascending=False)
        top_corr = [(i, j, corr.loc[i, j]) for (i, j) in corr_pairs.index if i != j][:5]
        analysis['top_correlations'] = top_corr
        # Skewness & kurtosis
        analysis['skewness'] = data.skew(numeric_only=True).to_dict()
        analysis['kurtosis'] = data.kurtosis(numeric_only=True).to_dict()
        # Class distribution (if last column is categorical)
        target_col = data.columns[-1]
        if data[target_col].dtype == 'object' or len(data[target_col].unique()) < 20:
            analysis['class_distribution'] = data[target_col].value_counts().to_dict()
            analysis['imbalance'] = (data[target_col].value_counts(normalize=True).max() > 0.7)
        # Per-column summary
        col_summary = {}
        for col in data.columns:
            col_summary[col] = {
                'type': str(data[col].dtype),
                'missing': int(data[col].isnull().sum()),
                'unique': int(data[col].nunique()),
                'sample_values': data[col].dropna().unique()[:5].tolist()
            }
        analysis['column_summary'] = col_summary
        return analysis