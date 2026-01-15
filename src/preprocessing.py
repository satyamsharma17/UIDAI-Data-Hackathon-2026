"""
Data Preprocessing Module for UIDAI Data Hackathon 2026
Handles data cleaning, validation, and transformation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class AadhaarDataPreprocessor:
    """Preprocess Aadhaar datasets for analysis"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.validation_results = {}
        
    def validate_data(self, df: pd.DataFrame, dataset_type: str) -> Dict:
        """
        Validate data quality
        
        Args:
            df: Input DataFrame
            dataset_type: 'enrolment', 'demographic', or 'biometric'
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'date_range': None,
            'unique_states': None,
            'unique_districts': None,
            'unique_pincodes': None,
            'negative_values': {}
        }
        
        # Date validation
        if 'date' in df.columns:
            try:
                date_col = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
                results['date_range'] = (date_col.min(), date_col.max())
                results['invalid_dates'] = date_col.isnull().sum()
            except:
                results['invalid_dates'] = 'Unable to parse dates'
        
        # Geographic validation
        if 'state' in df.columns:
            results['unique_states'] = df['state'].nunique()
        if 'district' in df.columns:
            results['unique_districts'] = df['district'].nunique()
        if 'pincode' in df.columns:
            results['unique_pincodes'] = df['pincode'].nunique()
            # Check for invalid pincodes (should be 6 digits)
            results['invalid_pincodes'] = df[df['pincode'].astype(str).str.len() != 6].shape[0]
        
        # Check for negative values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                results['negative_values'][col] = neg_count
        
        self.validation_results[dataset_type] = results
        return results
    
    def clean_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Clean and standardize data
        
        Args:
            df: Input DataFrame
            dataset_type: 'enrolment', 'demographic', or 'biometric'
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Parse dates
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'], format='%d-%m-%Y', errors='coerce')
            # Remove rows with invalid dates
            df_clean = df_clean.dropna(subset=['date'])
        
        # Standardize state names (title case)
        if 'state' in df_clean.columns:
            df_clean['state'] = df_clean['state'].str.strip().str.title()
        
        # Standardize district names
        if 'district' in df_clean.columns:
            df_clean['district'] = df_clean['district'].str.strip().str.title()
        
        # Clean pincode (ensure 6 digits)
        if 'pincode' in df_clean.columns:
            df_clean['pincode'] = df_clean['pincode'].astype(str).str.zfill(6)
            # Remove invalid pincodes
            df_clean = df_clean[df_clean['pincode'].str.len() == 6]
            df_clean = df_clean[df_clean['pincode'].str.isdigit()]
        
        # Remove negative values in count columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean = df_clean[df_clean[col] >= 0]
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        return df_clean
    
    def add_derived_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Add derived features for analysis
        
        Args:
            df: Input DataFrame
            dataset_type: 'enrolment', 'demographic', or 'biometric'
            
        Returns:
            DataFrame with derived features
        """
        df_enhanced = df.copy()
        
        # Temporal features
        if 'date' in df_enhanced.columns:
            df_enhanced['year'] = df_enhanced['date'].dt.year
            df_enhanced['month'] = df_enhanced['date'].dt.month
            df_enhanced['quarter'] = df_enhanced['date'].dt.quarter
            df_enhanced['day_of_week'] = df_enhanced['date'].dt.dayofweek
            df_enhanced['week_of_year'] = df_enhanced['date'].dt.isocalendar().week
            df_enhanced['month_name'] = df_enhanced['date'].dt.month_name()
        
        # Total counts
        if dataset_type == 'enrolment':
            if all(col in df_enhanced.columns for col in ['age_0_5', 'age_5_17', 'age_18_greater']):
                df_enhanced['total_enrolments'] = (
                    df_enhanced['age_0_5'] + 
                    df_enhanced['age_5_17'] + 
                    df_enhanced['age_18_greater']
                )
                # Age group proportions
                df_enhanced['prop_age_0_5'] = df_enhanced['age_0_5'] / df_enhanced['total_enrolments']
                df_enhanced['prop_age_5_17'] = df_enhanced['age_5_17'] / df_enhanced['total_enrolments']
                df_enhanced['prop_age_18_greater'] = df_enhanced['age_18_greater'] / df_enhanced['total_enrolments']
        
        elif dataset_type == 'demographic':
            if all(col in df_enhanced.columns for col in ['demo_age_5_17', 'demo_age_17_']):
                df_enhanced['total_demo_updates'] = (
                    df_enhanced['demo_age_5_17'] + 
                    df_enhanced['demo_age_17_']
                )
                df_enhanced['prop_demo_youth'] = df_enhanced['demo_age_5_17'] / df_enhanced['total_demo_updates']
                df_enhanced['prop_demo_adult'] = df_enhanced['demo_age_17_'] / df_enhanced['total_demo_updates']
        
        elif dataset_type == 'biometric':
            if all(col in df_enhanced.columns for col in ['bio_age_5_17', 'bio_age_17_']):
                df_enhanced['total_bio_updates'] = (
                    df_enhanced['bio_age_5_17'] + 
                    df_enhanced['bio_age_17_']
                )
                df_enhanced['prop_bio_youth'] = df_enhanced['bio_age_5_17'] / df_enhanced['total_bio_updates']
                df_enhanced['prop_bio_adult'] = df_enhanced['bio_age_17_'] / df_enhanced['total_bio_updates']
        
        # Geographic hierarchy
        if 'state' in df_enhanced.columns and 'district' in df_enhanced.columns:
            df_enhanced['state_district'] = df_enhanced['state'] + ' - ' + df_enhanced['district']
        
        return df_enhanced
    
    def aggregate_by_geography(
        self, 
        df: pd.DataFrame, 
        level: str = 'state'
    ) -> pd.DataFrame:
        """
        Aggregate data by geographic level
        
        Args:
            df: Input DataFrame
            level: 'state', 'district', or 'pincode'
            
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
        
        # Get numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        agg_df = df.groupby(group_cols)[numeric_cols].sum().reset_index()
        
        return agg_df
    
    def aggregate_by_time(
        self, 
        df: pd.DataFrame, 
        freq: str = 'M'
    ) -> pd.DataFrame:
        """
        Aggregate data by time period
        
        Args:
            df: Input DataFrame
            freq: Pandas frequency string ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Time-aggregated DataFrame
        """
        if 'date' not in df.columns:
            raise ValueError("DataFrame must have 'date' column")
        
        df_copy = df.copy()
        df_copy = df_copy.set_index('date')
        
        # Get numeric columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        # Resample and aggregate
        agg_df = df_copy[numeric_cols].resample(freq).sum().reset_index()
        
        return agg_df
    
    def merge_datasets(
        self, 
        enrolment_df: pd.DataFrame, 
        demographic_df: Optional[pd.DataFrame] = None,
        biometric_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge enrolment, demographic, and biometric datasets
        
        Args:
            enrolment_df: Enrolment DataFrame
            demographic_df: Demographic update DataFrame (optional)
            biometric_df: Biometric update DataFrame (optional)
            
        Returns:
            Merged DataFrame
        """
        merge_keys = ['date', 'state', 'district', 'pincode']
        
        merged_df = enrolment_df.copy()
        
        if demographic_df is not None:
            merged_df = merged_df.merge(
                demographic_df, 
                on=merge_keys, 
                how='outer',
                suffixes=('', '_demo')
            )
        
        if biometric_df is not None:
            merged_df = merged_df.merge(
                biometric_df, 
                on=merge_keys, 
                how='outer',
                suffixes=('', '_bio')
            )
        
        # Fill NaN values with 0 for count columns
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
        
        return merged_df
    
    def get_summary_statistics(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Generate summary statistics
        
        Args:
            df: Input DataFrame
            dataset_type: 'enrolment', 'demographic', or 'biometric'
            
        Returns:
            Summary statistics DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = df[numeric_cols].describe()
        
        return summary


if __name__ == "__main__":
    # Example usage
    from data_loader import AadhaarDataLoader, parse_date_column
    
    loader = AadhaarDataLoader('/Users/satyamsharma/Satverse AI/UIDAI Data Hackathon 2026')
    preprocessor = AadhaarDataPreprocessor()
    
    # Load sample data
    print("Loading enrolment data sample...")
    enrolment_df = loader.load_dataset('enrolment', sample_frac=0.1)
    
    # Validate
    print("\nValidating data...")
    validation = preprocessor.validate_data(enrolment_df, 'enrolment')
    print(f"Total rows: {validation['total_rows']:,}")
    print(f"Duplicate rows: {validation['duplicate_rows']:,}")
    print(f"Date range: {validation['date_range']}")
    
    # Clean
    print("\nCleaning data...")
    enrolment_clean = preprocessor.clean_data(enrolment_df, 'enrolment')
    print(f"Rows after cleaning: {len(enrolment_clean):,}")
    
    # Add features
    print("\nAdding derived features...")
    enrolment_enhanced = preprocessor.add_derived_features(enrolment_clean, 'enrolment')
    print(f"Columns: {enrolment_enhanced.columns.tolist()}")
    
    # Summary statistics
    print("\nSummary statistics:")
    print(preprocessor.get_summary_statistics(enrolment_enhanced, 'enrolment'))
