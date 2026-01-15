"""
Anomaly Detection Module for UIDAI Data Hackathon 2026
Statistical and ML-based anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Detect anomalies in Aadhaar data"""
    
    def __init__(self):
        """Initialize anomaly detector"""
        self.scaler = StandardScaler()
    
    def detect_statistical_outliers(
        self,
        df: pd.DataFrame,
        value_col: str,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect statistical outliers
        
        Args:
            df: Input DataFrame
            value_col: Column to analyze
            method: 'iqr', 'zscore', or 'modified_zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier flag
        """
        df_copy = df.copy()
        
        if method == 'iqr':
            Q1 = df_copy[value_col].quantile(0.25)
            Q3 = df_copy[value_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_copy['is_outlier'] = (
                (df_copy[value_col] < lower_bound) | 
                (df_copy[value_col] > upper_bound)
            )
            df_copy['outlier_score'] = np.abs(df_copy[value_col] - df_copy[value_col].median()) / IQR
        
        elif method == 'zscore':
            mean = df_copy[value_col].mean()
            std = df_copy[value_col].std()
            
            df_copy['zscore'] = (df_copy[value_col] - mean) / std
            df_copy['is_outlier'] = np.abs(df_copy['zscore']) > threshold
            df_copy['outlier_score'] = np.abs(df_copy['zscore'])
        
        elif method == 'modified_zscore':
            median = df_copy[value_col].median()
            mad = np.median(np.abs(df_copy[value_col] - median))
            
            if mad == 0:
                df_copy['modified_zscore'] = 0
            else:
                df_copy['modified_zscore'] = 0.6745 * (df_copy[value_col] - median) / mad
            
            df_copy['is_outlier'] = np.abs(df_copy['modified_zscore']) > threshold
            df_copy['outlier_score'] = np.abs(df_copy['modified_zscore'])
        
        else:
            raise ValueError("method must be 'iqr', 'zscore', or 'modified_zscore'")
        
        return df_copy
    
    def detect_temporal_anomalies(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        window: int = 7,
        threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Detect temporal anomalies using moving statistics
        
        Args:
            df: Input DataFrame
            date_col: Date column
            value_col: Value column
            window: Window size for moving statistics
            threshold: Number of standard deviations
            
        Returns:
            DataFrame with anomaly flags
        """
        # Aggregate by date
        df_agg = df.groupby(date_col)[value_col].sum().reset_index()
        df_agg = df_agg.sort_values(date_col)
        
        # Calculate moving statistics
        df_agg['moving_mean'] = df_agg[value_col].rolling(window=window, center=True).mean()
        df_agg['moving_std'] = df_agg[value_col].rolling(window=window, center=True).std()
        
        # Detect anomalies
        df_agg['lower_bound'] = df_agg['moving_mean'] - threshold * df_agg['moving_std']
        df_agg['upper_bound'] = df_agg['moving_mean'] + threshold * df_agg['moving_std']
        
        df_agg['is_anomaly'] = (
            (df_agg[value_col] < df_agg['lower_bound']) | 
            (df_agg[value_col] > df_agg['upper_bound'])
        )
        
        # Calculate anomaly score
        df_agg['anomaly_score'] = np.abs(
            df_agg[value_col] - df_agg['moving_mean']
        ) / df_agg['moving_std']
        
        return df_agg
    
    def detect_multivariate_anomalies(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        contamination: float = 0.1
    ) -> pd.DataFrame:
        """
        Detect multivariate anomalies using Isolation Forest
        
        Args:
            df: Input DataFrame
            feature_cols: Columns to use for detection
            contamination: Expected proportion of outliers
            
        Returns:
            DataFrame with anomaly predictions
        """
        df_copy = df.copy()
        
        # Prepare features
        X = df_copy[feature_cols].fillna(0)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Predict anomalies (-1 for anomalies, 1 for normal)
        df_copy['anomaly_prediction'] = iso_forest.fit_predict(X_scaled)
        df_copy['is_anomaly_ml'] = df_copy['anomaly_prediction'] == -1
        
        # Get anomaly scores
        df_copy['anomaly_score_ml'] = -iso_forest.score_samples(X_scaled)
        
        return df_copy
    
    def detect_changepoints(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        min_size: int = 7
    ) -> List[pd.Timestamp]:
        """
        Detect changepoints in time series
        
        Args:
            df: Input DataFrame
            date_col: Date column
            value_col: Value column
            min_size: Minimum segment size
            
        Returns:
            List of changepoint dates
        """
        # Aggregate by date
        df_agg = df.groupby(date_col)[value_col].sum().reset_index()
        df_agg = df_agg.sort_values(date_col)
        
        values = df_agg[value_col].values
        dates = df_agg[date_col].values
        
        changepoints = []
        
        # Simple CUSUM-based changepoint detection
        mean = np.mean(values)
        std = np.std(values)
        
        cusum_pos = 0
        cusum_neg = 0
        threshold = 3 * std
        
        for i in range(len(values)):
            cusum_pos = max(0, cusum_pos + (values[i] - mean))
            cusum_neg = min(0, cusum_neg + (values[i] - mean))
            
            if cusum_pos > threshold or cusum_neg < -threshold:
                if i >= min_size and (len(changepoints) == 0 or i - changepoints[-1] >= min_size):
                    changepoints.append(i)
                cusum_pos = 0
                cusum_neg = 0
        
        # Convert indices to dates
        changepoint_dates = [dates[i] for i in changepoints]
        
        return changepoint_dates
    
    def analyze_geographic_anomalies(
        self,
        df: pd.DataFrame,
        geo_col: str,
        value_col: str,
        method: str = 'iqr'
    ) -> pd.DataFrame:
        """
        Detect anomalies at geographic level
        
        Args:
            df: Input DataFrame
            geo_col: Geographic column
            value_col: Value column
            method: Detection method
            
        Returns:
            DataFrame with geographic anomalies
        """
        # Aggregate by geography
        df_agg = df.groupby(geo_col)[value_col].sum().reset_index()
        
        # Detect outliers
        df_anomalies = self.detect_statistical_outliers(df_agg, value_col, method=method)
        
        # Filter to anomalies only
        anomalies = df_anomalies[df_anomalies['is_outlier']].copy()
        anomalies = anomalies.sort_values('outlier_score', ascending=False)
        
        return anomalies
    
    def generate_anomaly_report(
        self,
        df: pd.DataFrame,
        anomaly_col: str = 'is_anomaly'
    ) -> Dict:
        """
        Generate summary report of anomalies
        
        Args:
            df: DataFrame with anomaly flags
            anomaly_col: Column indicating anomalies
            
        Returns:
            Dictionary with anomaly statistics
        """
        total_records = len(df)
        num_anomalies = df[anomaly_col].sum()
        anomaly_rate = num_anomalies / total_records if total_records > 0 else 0
        
        report = {
            'total_records': total_records,
            'num_anomalies': num_anomalies,
            'anomaly_rate': anomaly_rate,
            'percentage': anomaly_rate * 100
        }
        
        # Add statistics by category if available
        if 'state' in df.columns:
            state_anomalies = df[df[anomaly_col]].groupby('state').size().sort_values(ascending=False)
            report['top_anomalous_states'] = state_anomalies.head(10).to_dict()
        
        if 'date' in df.columns:
            date_anomalies = df[df[anomaly_col]].groupby('date').size().sort_values(ascending=False)
            report['top_anomalous_dates'] = date_anomalies.head(10).to_dict()
        
        return report
    
    def visualize_anomalies_summary(
        self,
        df: pd.DataFrame,
        anomaly_col: str = 'is_anomaly'
    ) -> str:
        """
        Create text summary of anomalies
        
        Args:
            df: DataFrame with anomalies
            anomaly_col: Anomaly column
            
        Returns:
            Summary string
        """
        report = self.generate_anomaly_report(df, anomaly_col)
        
        summary = f"""
ANOMALY DETECTION SUMMARY
{'='*80}
Total Records: {report['total_records']:,}
Anomalies Detected: {report['num_anomalies']:,}
Anomaly Rate: {report['percentage']:.2f}%

"""
        
        if 'top_anomalous_states' in report:
            summary += "Top Anomalous States:\n"
            for state, count in list(report['top_anomalous_states'].items())[:5]:
                summary += f"  - {state}: {count:,}\n"
        
        return summary


if __name__ == "__main__":
    print("Anomaly detection module loaded successfully")
