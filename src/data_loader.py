"""
Data Loader Module for UIDAI Data Hackathon 2026
Handles loading and merging of all Aadhaar dataset splits
"""

import pandas as pd
import glob
import os
from typing import List, Dict, Tuple
from tqdm import tqdm


class AadhaarDataLoader:
    """Load and merge Aadhaar enrolment, demographic, and biometric datasets"""
    
    def __init__(self, base_path: str):
        """
        Initialize data loader
        
        Args:
            base_path: Base directory containing data folders
        """
        self.base_path = base_path
        self.enrolment_path = os.path.join(base_path, 'api_data_aadhar_enrolment')
        self.demographic_path = os.path.join(base_path, 'api_data_aadhar_demographic')
        self.biometric_path = os.path.join(base_path, 'api_data_aadhar_biometric')
        
    def load_dataset(self, dataset_type: str, sample_frac: float = 1.0) -> pd.DataFrame:
        """
        Load all CSV files for a specific dataset type
        
        Args:
            dataset_type: 'enrolment', 'demographic', or 'biometric'
            sample_frac: Fraction of data to sample (1.0 = all data)
            
        Returns:
            Merged DataFrame
        """
        path_mapping = {
            'enrolment': self.enrolment_path,
            'demographic': self.demographic_path,
            'biometric': self.biometric_path
        }
        
        if dataset_type not in path_mapping:
            raise ValueError(f"Invalid dataset_type. Choose from: {list(path_mapping.keys())}")
        
        folder_path = path_mapping[dataset_type]
        csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
        
        print(f"Loading {dataset_type} dataset from {len(csv_files)} files...")
        
        dfs = []
        for file in tqdm(csv_files):
            df = pd.read_csv(file)
            if sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=42)
            dfs.append(df)
        
        merged_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(merged_df):,} records for {dataset_type}")
        
        return merged_df
    
    def load_all_datasets(self, sample_frac: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Load all three datasets
        
        Args:
            sample_frac: Fraction of data to sample
            
        Returns:
            Dictionary with keys 'enrolment', 'demographic', 'biometric'
        """
        datasets = {}
        
        for dataset_type in ['enrolment', 'demographic', 'biometric']:
            datasets[dataset_type] = self.load_dataset(dataset_type, sample_frac)
        
        return datasets
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """
        Get information about available datasets
        
        Returns:
            Dictionary with dataset statistics
        """
        info = {}
        
        for dataset_type in ['enrolment', 'demographic', 'biometric']:
            path_mapping = {
                'enrolment': self.enrolment_path,
                'demographic': self.demographic_path,
                'biometric': self.biometric_path
            }
            
            folder_path = path_mapping[dataset_type]
            csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
            
            total_rows = 0
            for file in csv_files:
                # Quick row count without loading full file
                with open(file, 'r') as f:
                    total_rows += sum(1 for _ in f) - 1  # Subtract header
            
            info[dataset_type] = {
                'num_files': len(csv_files),
                'total_rows': total_rows,
                'files': [os.path.basename(f) for f in csv_files]
            }
        
        return info


def parse_date_column(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Parse date column to datetime
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        
    Returns:
        DataFrame with parsed date column
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format='%d-%m-%Y')
    return df


def get_column_mapping() -> Dict[str, Dict[str, List[str]]]:
    """
    Get column mappings for each dataset
    
    Returns:
        Dictionary mapping dataset types to column information
    """
    return {
        'enrolment': {
            'geographic': ['state', 'district', 'pincode'],
            'temporal': ['date'],
            'age_groups': ['age_0_5', 'age_5_17', 'age_18_greater']
        },
        'demographic': {
            'geographic': ['state', 'district', 'pincode'],
            'temporal': ['date'],
            'age_groups': ['demo_age_5_17', 'demo_age_17_']
        },
        'biometric': {
            'geographic': ['state', 'district', 'pincode'],
            'temporal': ['date'],
            'age_groups': ['bio_age_5_17', 'bio_age_17_']
        }
    }


if __name__ == "__main__":
    # Example usage
    loader = AadhaarDataLoader('/Users/satyamsharma/Satverse AI/UIDAI Data Hackathon 2026')
    
    # Get dataset info
    info = loader.get_dataset_info()
    print("\nDataset Information:")
    for dataset_type, details in info.items():
        print(f"\n{dataset_type.upper()}:")
        print(f"  Files: {details['num_files']}")
        print(f"  Total rows: {details['total_rows']:,}")
    
    # Load sample of data
    print("\n\nLoading 10% sample of each dataset...")
    datasets = loader.load_all_datasets(sample_frac=0.1)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()} Sample:")
        print(df.head())
        print(f"Shape: {df.shape}")
