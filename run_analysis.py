"""
Main analysis script to run end-to-end workflow
Can be used for quick analysis or batch processing
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from data_loader import AadhaarDataLoader
from preprocessing import AadhaarDataPreprocessor
from visualization import AadhaarVisualizer
from temporal_analysis import TemporalAnalyzer
from spatial_analysis import SpatialAnalyzer
from anomaly_detector import AnomalyDetector
from forecasting import ForecastingEngine
import os


def main(sample_frac=0.2):
    """
    Run complete analysis workflow
    
    Args:
        sample_frac: Fraction of data to analyze (0.1-1.0)
    """
    print("="*80)
    print("UIDAI DATA HACKATHON 2026 - ANALYSIS WORKFLOW")
    print("="*80)
    
    # Initialize
    BASE_PATH = '/Users/satyamsharma/Satverse AI/UIDAI Data Hackathon 2026'
    loader = AadhaarDataLoader(BASE_PATH)
    preprocessor = AadhaarDataPreprocessor()
    
    # Load data
    print(f"\n1. Loading datasets ({sample_frac*100}% sample)...")
    datasets = loader.load_all_datasets(sample_frac=sample_frac)
    
    # Clean and preprocess
    print("\n2. Cleaning and preprocessing...")
    enrolment_clean = preprocessor.clean_data(datasets['enrolment'], 'enrolment')
    demographic_clean = preprocessor.clean_data(datasets['demographic'], 'demographic')
    biometric_clean = preprocessor.clean_data(datasets['biometric'], 'biometric')
    
    # Add features
    print("\n3. Feature engineering...")
    enrolment_enhanced = preprocessor.add_derived_features(enrolment_clean, 'enrolment')
    demographic_enhanced = preprocessor.add_derived_features(demographic_clean, 'demographic')
    biometric_enhanced = preprocessor.add_derived_features(biometric_clean, 'biometric')
    
    # Save processed data
    print("\n4. Saving processed data...")
    os.makedirs('outputs', exist_ok=True)
    enrolment_enhanced.to_parquet('outputs/enrolment_processed.parquet', index=False)
    demographic_enhanced.to_parquet('outputs/demographic_processed.parquet', index=False)
    biometric_enhanced.to_parquet('outputs/biometric_processed.parquet', index=False)
    
    # Quick analysis
    print("\n5. Quick Analysis Summary:")
    print(f"   Enrolment records: {len(enrolment_enhanced):,}")
    print(f"   Demographic update records: {len(demographic_enhanced):,}")
    print(f"   Biometric update records: {len(biometric_enhanced):,}")
    print(f"   Date range: {enrolment_enhanced['date'].min()} to {enrolment_enhanced['date'].max()}")
    print(f"   States covered: {enrolment_enhanced['state'].nunique()}")
    print(f"   Districts covered: {enrolment_enhanced['district'].nunique()}")
    
    print("\n6. Total Activity:")
    print(f"   Total enrolments: {enrolment_enhanced['total_enrolments'].sum():,}")
    print(f"   Total demographic updates: {demographic_enhanced['total_demo_updates'].sum():,}")
    print(f"   Total biometric updates: {biometric_enhanced['total_bio_updates'].sum():,}")
    
    print("\n✓ Workflow completed successfully")
    print("  Next: Run Jupyter notebooks for detailed analysis")
    print("="*80)


if __name__ == "__main__":
    # Adjust sample_frac as needed (0.1 = 10%, 1.0 = 100%)
    main(sample_frac=0.2)
