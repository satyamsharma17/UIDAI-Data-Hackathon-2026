"""
Cohort Analysis Module
Advanced segmentation and cohort behavior analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class CohortAnalyzer:
    """
    Analyzes cohort behavior patterns and transitions
    """
    
    def __init__(self):
        self.age_thresholds = {
            'children': 18,
            'adults': 18
        }
    
    def segment_by_age(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Segment data into children vs adults
        """
        children = df[df['age'] < self.age_thresholds['children']].copy()
        adults = df[df['age'] >= self.age_thresholds['adults']].copy()
        
        return {
            'children': children,
            'adults': adults
        }
    
    def compare_update_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Compare update patterns between children and adults
        """
        segments = self.segment_by_age(df) 
        
        for segment_name, segment_df in segments.items():
            if len(segment_df) == 0:
                continue
            
            # Update type distribution
            if 'update_type' in segment_df.columns:
                update_dist = segment_df['update_type'].value_counts(normalize=True)
            else:
                update_dist = pd.Series()
            
            # Temporal patterns
            segment_df['date'] = pd.to_datetime(segment_df['date'])
            segment_df['day_of_week'] = segment_df['date'].dt.day_name()
            segment_df['hour'] = segment_df['date'].dt.hour if segment_df['date'].dt.hour.notna().any() else None
            
            # Calculate metrics
            results[segment_name] = {
                'count': len(segment_df),
                'percentage': len(segment_df) / len(df) * 100,
                'avg_age': segment_df['age'].mean(),
                'median_age': segment_df['age'].median(),
                'update_type_distribution': update_dist.to_dict() if not update_dist.empty else {},
                'day_of_week_distribution': segment_df['day_of_week'].value_counts(normalize=True).to_dict(),
                'states_covered': segment_df['state_name'].nunique() if 'state_name' in segment_df.columns else 0,
                'districts_covered': segment_df['district_name'].nunique() if 'district_name' in segment_df.columns else 0
            }
        
        # Calculate differences
        if 'children' in results and 'adults' in results:
            results['differences'] = {
                'count_ratio': results['adults']['count'] / results['children']['count'] if results['children']['count'] > 0 else 0,
                'avg_age_diff': results['adults']['avg_age'] - results['children']['avg_age'],
                'geographic_coverage_diff': {
                    'states': results['adults']['states_covered'] - results['children']['states_covered'],
                    'districts': results['adults']['districts_covered'] - results['children']['districts_covered']
                }
            }
        
        return results
    
    def create_transition_matrix(self, df: pd.DataFrame, 
                                 from_col: str = 'age_group',
                                 to_col: str = 'update_type') -> pd.DataFrame:
        """
        Create transition matrix showing movement between states
        """
        # Create contingency table
        transition = pd.crosstab(df[from_col], df[to_col], normalize='index')
        
        return transition
    
    def analyze_biometric_demographic_correlation(self, 
                                                   demographic_df: pd.DataFrame,
                                                   biometric_df: pd.DataFrame) -> Dict:
        """
        Deep-dive correlation analysis between biometric and demographic updates
        """
        # Merge on common fields
        if 'aadhaar_number' in demographic_df.columns and 'aadhaar_number' in biometric_df.columns:
            # Find users who appear in both datasets
            common_users = set(demographic_df['aadhaar_number'].unique()).intersection(
                set(biometric_df['aadhaar_number'].unique())
            )
            
            demo_common = demographic_df[demographic_df['aadhaar_number'].isin(common_users)]
            bio_common = biometric_df[biometric_df['aadhaar_number'].isin(common_users)]
            
            analysis = {
                'total_demographic_users': demographic_df['aadhaar_number'].nunique(),
                'total_biometric_users': biometric_df['aadhaar_number'].nunique(),
                'common_users': len(common_users),
                'overlap_percentage': len(common_users) / demographic_df['aadhaar_number'].nunique() * 100,
                'demographic_only': demographic_df['aadhaar_number'].nunique() - len(common_users),
                'biometric_only': biometric_df['aadhaar_number'].nunique() - len(common_users)
            }
        else:
            # Aggregate analysis without user-level matching
            analysis = {
                'demographic_records': len(demographic_df),
                'biometric_records': len(biometric_df),
                'ratio': len(biometric_df) / len(demographic_df) if len(demographic_df) > 0 else 0,
                'note': 'User-level correlation not available without common identifier'
            }
        
        # State-level correlation
        if 'state_name' in demographic_df.columns and 'state_name' in biometric_df.columns:
            demo_by_state = demographic_df['state_name'].value_counts()
            bio_by_state = biometric_df['state_name'].value_counts()
            
            # Combine and calculate correlation
            combined = pd.DataFrame({
                'demographic': demo_by_state,
                'biometric': bio_by_state
            }).fillna(0)
            
            if len(combined) > 1:
                correlation = combined['demographic'].corr(combined['biometric'])
                analysis['state_level_correlation'] = correlation
        
        return analysis
    
    def analyze_update_journey(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the journey/flow of updates over time
        """
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date to track journey
        df_sorted = df.sort_values('date')
        
        # Calculate time between updates (if user ID available)
        journey_stats = {
            'total_updates': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'daily_average': len(df) / ((df['date'].max() - df['date'].min()).days + 1)
        }
        
        # Update type transitions
        if 'update_type' in df.columns and len(df) > 1:
            df_sorted['next_update_type'] = df_sorted['update_type'].shift(-1)
            transitions = df_sorted.groupby(['update_type', 'next_update_type']).size().reset_index(name='count')
            transitions = transitions[transitions['next_update_type'].notna()]
            
            journey_stats['update_type_transitions'] = transitions.to_dict('records')
        
        # State transitions (geographic movement)
        if 'state_name' in df.columns and len(df) > 1:
            df_sorted['next_state'] = df_sorted['state_name'].shift(-1)
            state_transitions = df_sorted[df_sorted['state_name'] != df_sorted['next_state']].copy()
            state_transitions = state_transitions[state_transitions['next_state'].notna()]
            
            journey_stats['geographic_transitions'] = {
                'total_state_changes': len(state_transitions),
                'percentage_with_state_change': len(state_transitions) / len(df) * 100 if len(df) > 0 else 0
            }
        
        return journey_stats
    
    def create_cohort_lifecycle(self, df: pd.DataFrame) -> Dict:
        """
        Create lifecycle analysis for different cohorts
        """
        df['date'] = pd.to_datetime(df['date'])
        
        # Define cohorts by first update period
        df['first_update_month'] = df.groupby('aadhaar_number')['date'].transform('min').dt.to_period('M') if 'aadhaar_number' in df.columns else df['date'].dt.to_period('M')
        
        # Cohort metrics
        cohort_analysis = {}
        
        for cohort_period in df['first_update_month'].unique():
            cohort_data = df[df['first_update_month'] == cohort_period]
            
            cohort_analysis[str(cohort_period)] = {
                'size': len(cohort_data),
                'avg_age': cohort_data['age'].mean() if 'age' in cohort_data.columns else None,
                'states_covered': cohort_data['state_name'].nunique() if 'state_name' in cohort_data.columns else 0,
                'update_types': cohort_data['update_type'].value_counts().to_dict() if 'update_type' in cohort_data.columns else {}
            }
        
        return cohort_analysis


class SegmentationEngine:
    """
    Advanced segmentation using multiple criteria
    """
    
    def __init__(self):
        self.segment_definitions = {}
    
    def define_segment(self, name: str, criteria: Dict):
        """
        Define a custom segment
        
        Args:
            name: Segment name
            criteria: Dictionary of column:value pairs for filtering
        """
        self.segment_definitions[name] = criteria
    
    def apply_segmentation(self, df: pd.DataFrame, segment_name: str) -> pd.DataFrame:
        """
        Apply segmentation criteria to dataframe
        """
        if segment_name not in self.segment_definitions:
            raise ValueError(f"Segment '{segment_name}' not defined")
        
        criteria = self.segment_definitions[segment_name]
        mask = pd.Series([True] * len(df))
        
        for col, condition in criteria.items():
            if isinstance(condition, tuple):
                # Range condition (min, max)
                mask &= (df[col] >= condition[0]) & (df[col] <= condition[1])
            elif isinstance(condition, list):
                # Multiple values
                mask &= df[col].isin(condition)
            else:
                # Single value
                mask &= df[col] == condition
        
        return df[mask]
    
    def create_rfm_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM-like segments for update behavior
        (Recency, Frequency, Magnitude)
        """
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate RFM metrics per user (if user ID available)
        if 'aadhaar_number' in df.columns:
            max_date = df['date'].max()
            
            rfm = df.groupby('aadhaar_number').agg({
                'date': lambda x: (max_date - x.max()).days,  # Recency
                'aadhaar_number': 'count',  # Frequency
            }).reset_index()
            
            rfm.columns = ['aadhaar_number', 'recency', 'frequency']
            
            # Create segments based on quartiles
            rfm['recency_score'] = pd.qcut(rfm['recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop')
            rfm['frequency_score'] = pd.qcut(rfm['frequency'], q=4, labels=[1, 2, 3, 4], duplicates='drop')
            
            # Combine scores
            rfm['rfm_segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)
            
            # Merge back to original data
            df_with_rfm = df.merge(rfm[['aadhaar_number', 'rfm_segment']], on='aadhaar_number', how='left')
            
            return df_with_rfm
        else:
            return df
    
    def multi_dimensional_segmentation(self, df: pd.DataFrame) -> Dict:
        """
        Create segments based on multiple dimensions
        """
        segments = {}
        
        # Age-based segments
        if 'age' in df.columns:
            segments['by_age'] = {
                'children (0-18)': df[df['age'] <= 18],
                'young_adults (19-35)': df[(df['age'] > 18) & (df['age'] <= 35)],
                'middle_aged (36-55)': df[(df['age'] > 35) & (df['age'] <= 55)],
                'seniors (56+)': df[df['age'] > 55]
            }
        
        # Geographic segments
        if 'state_name' in df.columns:
            top_states = df['state_name'].value_counts().head(5).index
            segments['by_geography'] = {
                'top_5_states': df[df['state_name'].isin(top_states)],
                'other_states': df[~df['state_name'].isin(top_states)]
            }
        
        # Temporal segments
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            segments['by_time'] = {
                'weekdays': df[df['day_of_week'] < 5],
                'weekends': df[df['day_of_week'] >= 5]
            }
        
        # Update type segments
        if 'update_type' in df.columns:
            segments['by_update_type'] = {
                update_type: df[df['update_type'] == update_type]
                for update_type in df['update_type'].unique()
            }
        
        # Calculate segment sizes
        segment_summary = {}
        for dimension, dimension_segments in segments.items():
            segment_summary[dimension] = {
                name: len(seg_df) 
                for name, seg_df in dimension_segments.items()
            }
        
        return segment_summary


if __name__ == "__main__":
    print("Cohort Analysis Module")
    print("=" * 60)
    print("Functions:")
    print("  • Children vs Adults Comparison")
    print("  • Transition Matrix Generation")
    print("  • Biometric-Demographic Correlation")
    print("  • Update Journey Flow Analysis")
    print("  • Cohort Lifecycle Tracking")
    print("  • RFM Segmentation")
    print("  • Multi-dimensional Segmentation")
    print("=" * 60)
