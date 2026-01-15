"""
Temporal Analysis Module for UIDAI Data Hackathon 2026
Time series analysis, seasonality, and trends
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


class TemporalAnalyzer:
    """Time series analysis for Aadhaar data"""
    
    def __init__(self):
        """Initialize temporal analyzer"""
        pass
    
    def aggregate_by_time(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_cols: List[str],
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Aggregate data by time frequency
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            value_cols: Columns to aggregate
            freq: Frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Time-aggregated DataFrame
        """
        df_copy = df.copy()
        df_copy = df_copy.set_index(date_col)
        
        agg_df = df_copy[value_cols].resample(freq).sum().reset_index()
        
        return agg_df
    
    def calculate_growth_rates(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        periods: int = 1
    ) -> pd.DataFrame:
        """
        Calculate period-over-period growth rates
        
        Args:
            df: Input DataFrame
            date_col: Date column
            value_col: Value column
            periods: Number of periods for growth calculation
            
        Returns:
            DataFrame with growth rates
        """
        df_copy = df.copy()
        df_copy = df_copy.sort_values(date_col)
        
        df_copy[f'{value_col}_growth'] = df_copy[value_col].pct_change(periods=periods) * 100
        df_copy[f'{value_col}_change'] = df_copy[value_col].diff(periods=periods)
        
        return df_copy
    
    def detect_seasonality(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        period: int = 7
    ) -> Dict:
        """
        Detect and decompose seasonality
        
        Args:
            df: Input DataFrame
            date_col: Date column
            value_col: Value column
            period: Seasonal period
            
        Returns:
            Dictionary with decomposition components
        """
        # Aggregate by date
        df_agg = df.groupby(date_col)[value_col].sum().reset_index()
        df_agg = df_agg.set_index(date_col)
        
        # Perform seasonal decomposition
        try:
            decomposition = seasonal_decompose(
                df_agg[value_col],
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )
            
            result = {
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'seasonal_strength': self._calculate_seasonal_strength(decomposition)
            }
            
            return result
        except Exception as e:
            print(f"Seasonality detection failed: {e}")
            return None
    
    def _calculate_seasonal_strength(self, decomposition) -> float:
        """Calculate strength of seasonality"""
        var_resid = np.var(decomposition.resid.dropna())
        var_detrend = np.var((decomposition.observed - decomposition.trend).dropna())
        
        if var_detrend == 0:
            return 0.0
        
        seasonal_strength = max(0, 1 - var_resid / var_detrend)
        return seasonal_strength
    
    def test_stationarity(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str
    ) -> Dict:
        """
        Test time series stationarity using Augmented Dickey-Fuller test
        
        Args:
            df: Input DataFrame
            date_col: Date column
            value_col: Value column
            
        Returns:
            Dictionary with test results
        """
        # Aggregate by date
        df_agg = df.groupby(date_col)[value_col].sum().reset_index()
        
        # Perform ADF test
        result = adfuller(df_agg[value_col].dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def calculate_moving_averages(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        windows: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        Calculate moving averages
        
        Args:
            df: Input DataFrame
            date_col: Date column
            value_col: Value column
            windows: List of window sizes
            
        Returns:
            DataFrame with moving averages
        """
        df_copy = df.copy()
        df_copy = df_copy.sort_values(date_col)
        
        for window in windows:
            df_copy[f'{value_col}_ma_{window}'] = df_copy[value_col].rolling(window=window).mean()
        
        return df_copy
    
    def identify_peaks_troughs(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        order: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Identify peaks and troughs in time series
        
        Args:
            df: Input DataFrame
            date_col: Date column
            value_col: Value column
            order: How many points on each side to compare
            
        Returns:
            Dictionary with peaks and troughs DataFrames
        """
        from scipy.signal import argrelextrema
        
        # Aggregate by date
        df_agg = df.groupby(date_col)[value_col].sum().reset_index()
        df_agg = df_agg.sort_values(date_col)
        
        values = df_agg[value_col].values
        
        # Find peaks
        peak_indices = argrelextrema(values, np.greater, order=order)[0]
        peaks = df_agg.iloc[peak_indices]
        
        # Find troughs
        trough_indices = argrelextrema(values, np.less, order=order)[0]
        troughs = df_agg.iloc[trough_indices]
        
        return {
            'peaks': peaks,
            'troughs': troughs
        }
    
    def calculate_temporal_statistics(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        groupby: str = 'month'
    ) -> pd.DataFrame:
        """
        Calculate statistics by temporal groups
        
        Args:
            df: Input DataFrame
            date_col: Date column
            value_col: Value column
            groupby: Temporal grouping ('month', 'quarter', 'day_of_week')
            
        Returns:
            Statistics DataFrame
        """
        df_copy = df.copy()
        
        if groupby not in df_copy.columns:
            if groupby == 'month':
                df_copy['month'] = pd.to_datetime(df_copy[date_col]).dt.month
            elif groupby == 'quarter':
                df_copy['quarter'] = pd.to_datetime(df_copy[date_col]).dt.quarter
            elif groupby == 'day_of_week':
                df_copy['day_of_week'] = pd.to_datetime(df_copy[date_col]).dt.dayofweek
        
        stats = df_copy.groupby(groupby)[value_col].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).reset_index()
        
        return stats


if __name__ == "__main__":
    print("Temporal analysis module loaded successfully")
