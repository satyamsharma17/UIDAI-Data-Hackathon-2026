"""
Spatial Analysis Module for UIDAI Data Hackathon 2026
Geographic analysis and regional patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SpatialAnalyzer:
    """Geographic and spatial analysis for Aadhaar data"""
    
    def __init__(self):
        """Initialize spatial analyzer"""
        # Indian state populations (2021 estimates in millions)
        self.state_populations = {
            'Uttar Pradesh': 241.0,
            'Maharashtra': 123.1,
            'Bihar': 128.0,
            'West Bengal': 100.0,
            'Madhya Pradesh': 85.3,
            'Tamil Nadu': 77.8,
            'Rajasthan': 81.0,
            'Karnataka': 67.6,
            'Gujarat': 70.0,
            'Andhra Pradesh': 53.0,
            'Odisha': 46.4,
            'Telangana': 39.0,
            'Kerala': 35.7,
            'Jharkhand': 38.0,
            'Assam': 35.6,
            'Punjab': 30.1,
            'Chhattisgarh': 29.4,
            'Haryana': 28.9,
            'Delhi': 19.0,
            'Jammu And Kashmir': 13.6,
            'Uttarakhand': 11.4,
            'Himachal Pradesh': 7.5,
            'Tripura': 4.2,
            'Meghalaya': 3.4,
            'Manipur': 3.3,
            'Nagaland': 2.2,
            'Goa': 1.6,
            'Arunachal Pradesh': 1.6,
            'Mizoram': 1.2,
            'Sikkim': 0.7
        }
    
    def aggregate_by_geography(
        self,
        df: pd.DataFrame,
        level: str,
        value_cols: List[str]
    ) -> pd.DataFrame:
        """
        Aggregate data by geographic level
        
        Args:
            df: Input DataFrame
            level: 'state', 'district', or 'pincode'
            value_cols: Columns to aggregate
            
        Returns:
            Aggregated DataFrame
        """
        group_cols = []
        
        if level == 'state':
            group_cols = ['state']
        elif level == 'district':
            group_cols = ['state', 'district']
        elif level == 'pincode':
            group_cols = ['state', 'district', 'pincode']
        else:
            raise ValueError("level must be 'state', 'district', or 'pincode'")
        
        agg_df = df.groupby(group_cols)[value_cols].sum().reset_index()
        
        return agg_df
    
    def calculate_rates_per_capita(
        self,
        df: pd.DataFrame,
        value_col: str,
        state_col: str = 'state'
    ) -> pd.DataFrame:
        """
        Calculate per capita rates using population data
        
        Args:
            df: Input DataFrame with state column
            value_col: Value column to normalize
            state_col: State column name
            
        Returns:
            DataFrame with per capita rates
        """
        df_copy = df.copy()
        
        # Map population
        df_copy['population_millions'] = df_copy[state_col].map(self.state_populations)
        
        # Calculate per 1000 residents
        df_copy[f'{value_col}_per_1000'] = (
            df_copy[value_col] / (df_copy['population_millions'] * 1000)
        )
        
        # Calculate per 100k residents
        df_copy[f'{value_col}_per_100k'] = (
            df_copy[value_col] / (df_copy['population_millions'] * 10)
        )
        
        return df_copy
    
    def identify_geographic_outliers(
        self,
        df: pd.DataFrame,
        geo_col: str,
        value_col: str,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Identify geographic outliers
        
        Args:
            df: Input DataFrame
            geo_col: Geographic column
            value_col: Value column
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers
        """
        df_agg = df.groupby(geo_col)[value_col].sum().reset_index()
        
        if method == 'iqr':
            Q1 = df_agg[value_col].quantile(0.25)
            Q3 = df_agg[value_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df_agg[
                (df_agg[value_col] < lower_bound) | 
                (df_agg[value_col] > upper_bound)
            ]
        
        elif method == 'zscore':
            mean = df_agg[value_col].mean()
            std = df_agg[value_col].std()
            
            df_agg['zscore'] = (df_agg[value_col] - mean) / std
            outliers = df_agg[np.abs(df_agg['zscore']) > threshold]
        
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")
        
        return outliers
    
    def calculate_regional_concentration(
        self,
        df: pd.DataFrame,
        geo_col: str,
        value_col: str
    ) -> Dict:
        """
        Calculate concentration metrics (Gini, Herfindahl)
        
        Args:
            df: Input DataFrame
            geo_col: Geographic column
            value_col: Value column
            
        Returns:
            Dictionary with concentration metrics
        """
        df_agg = df.groupby(geo_col)[value_col].sum().sort_values(ascending=False).reset_index()
        
        # Calculate Gini coefficient
        values = df_agg[value_col].values
        n = len(values)
        
        if n == 0 or values.sum() == 0:
            gini = 0
        else:
            sorted_values = np.sort(values)
            cumsum = np.cumsum(sorted_values)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
        
        # Calculate Herfindahl-Hirschman Index
        total = values.sum()
        if total == 0:
            hhi = 0
        else:
            shares = values / total
            hhi = np.sum(shares ** 2)
        
        # Top N concentration
        top_5_share = values[:5].sum() / total if total > 0 else 0
        top_10_share = values[:10].sum() / total if total > 0 else 0
        
        return {
            'gini_coefficient': gini,
            'herfindahl_index': hhi,
            'top_5_share': top_5_share,
            'top_10_share': top_10_share,
            'num_regions': n
        }
    
    def compare_regions(
        self,
        df: pd.DataFrame,
        geo_col: str,
        value_cols: List[str],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Compare top regions across multiple metrics
        
        Args:
            df: Input DataFrame
            geo_col: Geographic column
            value_cols: List of value columns to compare
            top_n: Number of top regions to show
            
        Returns:
            Comparison DataFrame
        """
        df_agg = df.groupby(geo_col)[value_cols].sum().reset_index()
        
        # Calculate total score (normalized sum)
        for col in value_cols:
            max_val = df_agg[col].max()
            if max_val > 0:
                df_agg[f'{col}_normalized'] = df_agg[col] / max_val
        
        normalized_cols = [f'{col}_normalized' for col in value_cols]
        df_agg['composite_score'] = df_agg[normalized_cols].mean(axis=1)
        
        # Get top regions
        top_regions = df_agg.nlargest(top_n, 'composite_score')
        
        return top_regions
    
    def analyze_geographic_disparities(
        self,
        df: pd.DataFrame,
        geo_col: str,
        value_col: str
    ) -> Dict:
        """
        Analyze disparities across geographic regions
        
        Args:
            df: Input DataFrame
            geo_col: Geographic column
            value_col: Value column
            
        Returns:
            Dictionary with disparity metrics
        """
        df_agg = df.groupby(geo_col)[value_col].sum().reset_index()
        
        values = df_agg[value_col]
        
        disparity = {
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'cv': values.std() / values.mean() if values.mean() > 0 else 0,
            'min': values.min(),
            'max': values.max(),
            'range': values.max() - values.min(),
            'q1': values.quantile(0.25),
            'q3': values.quantile(0.75),
            'iqr': values.quantile(0.75) - values.quantile(0.25)
        }
        
        return disparity
    
    def create_state_ranking(
        self,
        df: pd.DataFrame,
        value_col: str,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Create state ranking by metric
        
        Args:
            df: Input DataFrame
            value_col: Value column to rank by
            ascending: Ranking order
            
        Returns:
            Ranked DataFrame
        """
        state_agg = df.groupby('state')[value_col].sum().reset_index()
        state_agg = state_agg.sort_values(value_col, ascending=ascending)
        state_agg['rank'] = range(1, len(state_agg) + 1)
        
        return state_agg


if __name__ == "__main__":
    print("Spatial analysis module loaded successfully")
