"""
Innovative Metrics Module
Creates unique composite indices for Aadhaar ecosystem analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AadhaarHealthIndex:
    """
    Composite index measuring overall health of Aadhaar ecosystem
    Components: Coverage, Update Velocity, Data Quality, Accessibility
    """
    
    def __init__(self):
        self.weights = {
            'coverage': 0.30,
            'update_velocity': 0.25,
            'data_quality': 0.25,
            'accessibility': 0.20
        }
        
    def calculate_coverage_score(self, enrolment_df: pd.DataFrame, 
                                 demographic_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate coverage metrics by state/district
        Higher score = better penetration
        """
        # State-level coverage
        state_coverage = enrolment_df.groupby('state').size()
        state_scores = (state_coverage - state_coverage.min()) / (state_coverage.max() - state_coverage.min())
        
        # District-level coverage
        district_coverage = enrolment_df.groupby('district').size()
        district_scores = (district_coverage - district_coverage.min()) / (district_coverage.max() - district_coverage.min())
        
        return {
            'state_scores': state_scores.to_dict(),
            'district_scores': district_scores.to_dict(),
            'overall_coverage': state_scores.mean() * 100
        }
    
    def calculate_update_velocity(self, demographic_df: pd.DataFrame,
                                  biometric_df: pd.DataFrame) -> Dict[str, float]:
        """
        Measure update activity rate (updates per day/month)
        Higher velocity = more active maintenance
        """
        # Daily update rates
        demo_daily = demographic_df.groupby('date').size()
        bio_daily = biometric_df.groupby('date').size()
        
        # Calculate velocity metrics
        demo_velocity = demo_daily.mean()
        bio_velocity = bio_daily.mean()
        
        # Normalize to 0-100 scale
        max_velocity = max(demo_velocity, bio_velocity)
        
        return {
            'demographic_velocity': (demo_velocity / max_velocity) * 100,
            'biometric_velocity': (bio_velocity / max_velocity) * 100,
            'overall_velocity': ((demo_velocity + bio_velocity) / (2 * max_velocity)) * 100
        }
    
    def calculate_data_quality_score(self, enrolment_df: pd.DataFrame,
                                     demographic_df: pd.DataFrame,
                                     biometric_df: pd.DataFrame) -> Dict[str, float]:
        """
        Assess data quality: completeness, validity, consistency
        """
        scores = {}
        
        # Completeness score (% non-null values)
        for name, df in [('enrolment', enrolment_df), 
                        ('demographic', demographic_df),
                        ('biometric', biometric_df)]:
            completeness = (1 - df.isnull().sum() / len(df)).mean() * 100
            scores[f'{name}_completeness'] = completeness
        
        # Validity score (% valid dates, positive numbers)
        validity_checks = []
        
        # Check date validity
        if 'date' in enrolment_df.columns:
            valid_dates = pd.to_datetime(enrolment_df['date'], errors='coerce').notna().mean()
            validity_checks.append(valid_dates)
        
        # Check positive values
        numeric_cols = enrolment_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'total' in col.lower() or 'count' in col.lower():
                valid_positive = (enrolment_df[col] >= 0).mean()
                validity_checks.append(valid_positive)
        
        scores['validity'] = np.mean(validity_checks) * 100 if validity_checks else 90.0
        scores['overall_quality'] = np.mean([v for k, v in scores.items()]) 
        
        return scores
    
    def calculate_accessibility_score(self, enrolment_df: pd.DataFrame) -> Dict[str, float]:
        """
        Measure service accessibility: geographic spread, temporal availability
        """
        # Geographic spread (Gini coefficient inverse)
        state_totals = enrolment_df.groupby('state')['total_enrolments'].sum().values
        sorted_totals = np.sort(state_totals)
        n = len(sorted_totals)
        cumsum = np.cumsum(sorted_totals)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_totals)) / (n * np.sum(sorted_totals)) - (n + 1) / n
        
        # Inverse Gini (0 = high inequality, 1 = perfect equality)
        equality_score = (1 - gini) * 100
        
        # Temporal availability (days with service)
        total_days = (enrolment_df['date'].max() - enrolment_df['date'].min()).days
        active_days = enrolment_df['date'].nunique()
        temporal_score = (active_days / total_days) * 100 if total_days > 0 else 100
        
        return {
            'geographic_equality': equality_score,
            'temporal_availability': temporal_score,
            'overall_accessibility': (equality_score + temporal_score) / 2
        }
    
    def calculate_composite_index(self, enrolment_df: pd.DataFrame,
                                  demographic_df: pd.DataFrame,
                                  biometric_df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate overall Aadhaar Health Index (0-100)
        """
        # Calculate component scores
        coverage = self.calculate_coverage_score(enrolment_df, demographic_df)
        velocity = self.calculate_update_velocity(demographic_df, biometric_df)
        quality = self.calculate_data_quality_score(enrolment_df, demographic_df, biometric_df)
        accessibility = self.calculate_accessibility_score(enrolment_df)
        
        # Composite score
        ahi_score = (
            self.weights['coverage'] * coverage['overall_coverage'] +
            self.weights['update_velocity'] * velocity['overall_velocity'] +
            self.weights['data_quality'] * quality['overall_quality'] +
            self.weights['accessibility'] * accessibility['overall_accessibility']
        )
        
        # Rating
        if ahi_score >= 90:
            rating = "EXCELLENT"
        elif ahi_score >= 75:
            rating = "GOOD"
        elif ahi_score >= 60:
            rating = "FAIR"
        else:
            rating = "NEEDS IMPROVEMENT"
        
        return {
            'aadhaar_health_index': ahi_score,
            'rating': rating,
            'components': {
                'coverage': coverage['overall_coverage'],
                'update_velocity': velocity['overall_velocity'],
                'data_quality': quality['overall_quality'],
                'accessibility': accessibility['overall_accessibility']
            },
            'detailed_metrics': {
                'coverage': coverage,
                'velocity': velocity,
                'quality': quality,
                'accessibility': accessibility
            }
        }


class DigitalInclusionScore:
    """
    Measures digital inclusion across demographics and geographies
    Focuses on mobile/email updates as digital adoption proxy
    """
    
    def __init__(self):
        self.weights = {
            'digital_update_rate': 0.40,
            'demographic_parity': 0.30,
            'geographic_spread': 0.30
        }
    
    def calculate_digital_update_rate(self, demographic_df: pd.DataFrame) -> float:
        """
        Calculate percentage of digital updates (mobile/email)
        """
        digital_fields = ['mobile_update', 'email_update']
        
        digital_updates = 0
        for field in digital_fields:
            if field in demographic_df.columns:
                digital_updates += demographic_df[field].sum()
        
        total_updates = len(demographic_df)
        digital_rate = (digital_updates / total_updates) * 100 if total_updates > 0 else 0
        
        return digital_rate
    
    def calculate_demographic_parity(self, demographic_df: pd.DataFrame) -> Dict[str, float]:
        """
        Measure digital inclusion parity across age groups
        """
        if 'age_group' not in demographic_df.columns:
            return {'parity_score': 75.0}  # Default if age not available
        
        # Digital update rates by age group
        age_digital_rates = {}
        digital_fields = ['mobile_update', 'email_update']
        
        for age_group in demographic_df['age_group'].unique():
            age_data = demographic_df[demographic_df['age_group'] == age_group]
            digital_count = sum(age_data[field].sum() for field in digital_fields if field in age_data.columns)
            age_digital_rates[age_group] = (digital_count / len(age_data)) * 100 if len(age_data) > 0 else 0
        
        # Calculate parity (inverse of coefficient of variation)
        rates = list(age_digital_rates.values())
        if len(rates) > 0 and np.mean(rates) > 0:
            cv = np.std(rates) / np.mean(rates)
            parity_score = (1 - min(cv, 1)) * 100
        else:
            parity_score = 50.0
        
        return {
            'parity_score': parity_score,
            'age_group_rates': age_digital_rates
        }
    
    def calculate_geographic_spread(self, demographic_df: pd.DataFrame) -> float:
        """
        Measure digital update spread across states
        """
        digital_fields = ['mobile_update', 'email_update']
        
        state_digital_rates = {}
        for state in demographic_df['state'].unique():
            state_data = demographic_df[demographic_df['state'] == state]
            digital_count = sum(state_data[field].sum() for field in digital_fields if field in state_data.columns)
            state_digital_rates[state] = (digital_count / len(state_data)) * 100 if len(state_data) > 0 else 0
        
        # Calculate spread (% of states with >10% digital rate)
        states_with_digital = sum(1 for rate in state_digital_rates.values() if rate > 10)
        spread_score = (states_with_digital / len(state_digital_rates)) * 100 if len(state_digital_rates) > 0 else 0
        
        return spread_score
    
    def calculate_composite_score(self, demographic_df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate Digital Inclusion Score (0-100)
        """
        digital_rate = self.calculate_digital_update_rate(demographic_df)
        parity = self.calculate_demographic_parity(demographic_df)
        spread = self.calculate_geographic_spread(demographic_df)
        
        # Composite score
        dis_score = (
            self.weights['digital_update_rate'] * digital_rate +
            self.weights['demographic_parity'] * parity['parity_score'] +
            self.weights['geographic_spread'] * spread
        )
        
        # Classification
        if dis_score >= 80:
            classification = "HIGHLY INCLUSIVE"
        elif dis_score >= 60:
            classification = "MODERATELY INCLUSIVE"
        elif dis_score >= 40:
            classification = "DEVELOPING"
        else:
            classification = "EARLY STAGE"
        
        return {
            'digital_inclusion_score': dis_score,
            'classification': classification,
            'components': {
                'digital_update_rate': digital_rate,
                'demographic_parity': parity['parity_score'],
                'geographic_spread': spread
            },
            'details': {
                'age_group_rates': parity.get('age_group_rates', {})
            }
        }


class ServiceAccessibilityIndex:
    """
    Measures ease of access to Aadhaar services
    Factors: Wait time proxy, geographic coverage, service density
    """
    
    def __init__(self):
        self.weights = {
            'load_balance': 0.35,
            'geographic_coverage': 0.35,
            'service_density': 0.30
        }
    
    def calculate_load_balance(self, enrolment_df: pd.DataFrame) -> Dict[str, float]:
        """
        Measure load distribution (proxy for wait times)
        Lower CV = better balance = lower wait times
        """
        # Daily enrolment distribution
        daily_enrol = enrolment_df.groupby('date')['total_enrolments'].sum()
        
        # Coefficient of variation (lower is better)
        cv = daily_enrol.std() / daily_enrol.mean() if daily_enrol.mean() > 0 else 1
        
        # Convert to 0-100 score (inverse of CV)
        load_balance_score = (1 - min(cv, 1)) * 100
        
        # Peak vs average ratio
        peak_ratio = daily_enrol.max() / daily_enrol.mean() if daily_enrol.mean() > 0 else 1
        
        return {
            'load_balance_score': load_balance_score,
            'coefficient_variation': cv,
            'peak_average_ratio': peak_ratio
        }
    
    def calculate_geographic_coverage(self, enrolment_df: pd.DataFrame) -> float:
        """
        Measure geographic penetration
        """
        # Unique districts served
        total_districts = enrolment_df['district'].nunique()
        
        # Assume target is 700+ districts nationally
        coverage_pct = min((total_districts / 700) * 100, 100)
        
        return coverage_pct
    
    def calculate_service_density(self, enrolment_df: pd.DataFrame) -> Dict[str, float]:
        """
        Measure service points per capita (estimated)
        """
        # Calculate enrolments per district per day
        district_daily_avg = enrolment_df.groupby('district')['total_enrolments'].mean()
        
        # Normalize to 0-100 (higher = better density)
        if len(district_daily_avg) > 0:
            density_score = min((district_daily_avg.mean() / 100) * 100, 100)
        else:
            density_score = 50.0
        
        return {
            'service_density_score': density_score,
            'avg_enrolments_per_district': district_daily_avg.mean() if len(district_daily_avg) > 0 else 0
        }
    
    def calculate_composite_index(self, enrolment_df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate Service Accessibility Index (0-100)
        """
        load_balance = self.calculate_load_balance(enrolment_df)
        coverage = self.calculate_geographic_coverage(enrolment_df)
        density = self.calculate_service_density(enrolment_df)
        
        # Composite score
        sai_score = (
            self.weights['load_balance'] * load_balance['load_balance_score'] +
            self.weights['geographic_coverage'] * coverage +
            self.weights['service_density'] * density['service_density_score']
        )
        
        # Rating
        if sai_score >= 85:
            rating = "HIGHLY ACCESSIBLE"
        elif sai_score >= 70:
            rating = "ACCESSIBLE"
        elif sai_score >= 55:
            rating = "MODERATELY ACCESSIBLE"
        else:
            rating = "NEEDS IMPROVEMENT"
        
        return {
            'service_accessibility_index': sai_score,
            'rating': rating,
            'components': {
                'load_balance': load_balance['load_balance_score'],
                'geographic_coverage': coverage,
                'service_density': density['service_density_score']
            },
            'detailed_metrics': {
                'load_balance': load_balance,
                'coverage': coverage,
                'density': density
            }
        }


def calculate_all_indices(enrolment_df: pd.DataFrame,
                         demographic_df: pd.DataFrame,
                         biometric_df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate all innovative indices in one call
    """
    ahi = AadhaarHealthIndex()
    dis = DigitalInclusionScore()
    sai = ServiceAccessibilityIndex()
    
    results = {
        'aadhaar_health_index': ahi.calculate_composite_index(enrolment_df, demographic_df, biometric_df),
        'digital_inclusion_score': dis.calculate_composite_score(demographic_df),
        'service_accessibility_index': sai.calculate_composite_index(enrolment_df)
    }
    
    return results


if __name__ == "__main__":
    print("Innovative Metrics Module")
    print("=" * 60)
    print("Available Indices:")
    print("  1. Aadhaar Health Index (AHI)")
    print("  2. Digital Inclusion Score (DIS)")
    print("  3. Service Accessibility Index (SAI)")
    print("=" * 60)
