"""
Advanced Statistics Module
Provides trivariate analysis, significance testing, and interaction effects
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, pearsonr
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TrivariateAnalyzer:
    """
    Performs three-way interactions and multi-dimensional analysis
    """
    
    def __init__(self):
        self.results = {}
    
    def analyze_age_geography_time(self, df: pd.DataFrame,
                                   value_col: str = 'total_enrolments') -> Dict:
        """
        Analyze Age × Geography × Time interactions
        """
        results = {}
        
        # Check required columns
        required = ['age_group', 'state', 'date']
        if not all(col in df.columns for col in required):
            return {'error': 'Missing required columns for trivariate analysis'}
        
        # 1. Three-way aggregation
        trivariate = df.groupby(['age_group', 'state', pd.Grouper(key='date', freq='M')])[value_col].sum()
        trivariate_df = trivariate.reset_index()
        results['trivariate_data'] = trivariate_df
        
        # 2. Interaction strength (variance explained by each factor)
        age_variance = df.groupby('age_group')[value_col].var().mean()
        state_variance = df.groupby('state')[value_col].var().mean()
        time_variance = df.groupby(pd.Grouper(key='date', freq='M'))[value_col].var().mean()
        
        total_variance = age_variance + state_variance + time_variance
        
        results['variance_decomposition'] = {
            'age_contribution': (age_variance / total_variance) * 100 if total_variance > 0 else 0,
            'geography_contribution': (state_variance / total_variance) * 100 if total_variance > 0 else 0,
            'time_contribution': (time_variance / total_variance) * 100 if total_variance > 0 else 0
        }
        
        # 3. Top combinations
        top_combinations = trivariate.nlargest(10)
        results['top_combinations'] = top_combinations.to_dict()
        
        # 4. Interaction patterns
        # Age-State interaction
        age_state_pivot = df.pivot_table(
            index='age_group',
            columns='state',
            values=value_col,
            aggfunc='mean'
        )
        results['age_state_correlation'] = age_state_pivot.corr().mean().mean()
        
        return results
    
    def analyze_update_type_demographics_geography(self, demographic_df: pd.DataFrame) -> Dict:
        """
        Analyze Update Type × Demographics × Geography
        """
        results = {}
        
        # Update fields
        update_fields = ['name_update', 'address_update', 'dob_update', 
                        'mobile_update', 'email_update']
        
        # Filter available fields
        available_fields = [f for f in update_fields if f in demographic_df.columns]
        
        if not available_fields:
            return {'error': 'No update fields found'}
        
        # Three-way analysis
        for field in available_fields:
            # State × Age × Update Type
            if 'age_group' in demographic_df.columns:
                state_age_update = demographic_df.groupby(['state', 'age_group'])[field].sum()
                results[f'{field}_by_state_age'] = state_age_update.to_dict()
        
        # Calculate interaction strength
        interaction_scores = {}
        for field in available_fields:
            # Variance by state
            state_var = demographic_df.groupby('state')[field].var().mean()
            
            # Variance by age (if available)
            if 'age_group' in demographic_df.columns:
                age_var = demographic_df.groupby('age_group')[field].var().mean()
                interaction_scores[field] = {
                    'state_variance': state_var,
                    'age_variance': age_var,
                    'interaction_strength': min(state_var, age_var) / max(state_var, age_var) if max(state_var, age_var) > 0 else 0
                }
        
        results['interaction_scores'] = interaction_scores
        
        return results
    
    def generate_3d_coordinates(self, df: pd.DataFrame,
                               x_col: str, y_col: str, z_col: str) -> Dict:
        """
        Generate 3D plot coordinates for visualization
        """
        # Aggregate if needed
        if df[x_col].dtype == 'object' or df[y_col].dtype == 'object':
            # Categorical encoding
            x_encoded = pd.Categorical(df[x_col]).codes
            y_encoded = pd.Categorical(df[y_col]).codes
            z_values = df[z_col].values
        else:
            x_encoded = df[x_col].values
            y_encoded = df[y_col].values
            z_values = df[z_col].values
        
        return {
            'x': x_encoded.tolist(),
            'y': y_encoded.tolist(),
            'z': z_values.tolist(),
            'x_labels': df[x_col].unique().tolist() if df[x_col].dtype == 'object' else None,
            'y_labels': df[y_col].unique().tolist() if df[y_col].dtype == 'object' else None
        }


class SignificanceTest:
    """
    Statistical significance testing suite
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def chi_square_test(self, df: pd.DataFrame, col1: str, col2: str) -> Dict:
        """
        Chi-square test for independence between categorical variables
        """
        # Create contingency table
        contingency = pd.crosstab(df[col1], df[col2])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Effect size (Cramér's V)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0
        
        return {
            'test': 'Chi-Square',
            'statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'is_significant': p_value < self.alpha,
            'effect_size_cramers_v': cramers_v,
            'interpretation': self._interpret_cramers_v(cramers_v)
        }
    
    def anova_test(self, df: pd.DataFrame, categorical_col: str, 
                   numeric_col: str) -> Dict:
        """
        One-way ANOVA test for differences across groups
        """
        groups = [group[numeric_col].values for name, group in df.groupby(categorical_col)]
        
        # Perform ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        # Effect size (eta-squared)
        ss_between = sum(len(g) * (np.mean(g) - df[numeric_col].mean())**2 for g in groups)
        ss_total = sum((df[numeric_col] - df[numeric_col].mean())**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'test': 'One-Way ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'effect_size_eta_squared': eta_squared,
            'interpretation': self._interpret_eta_squared(eta_squared),
            'group_means': {name: group[numeric_col].mean() 
                          for name, group in df.groupby(categorical_col)}
        }
    
    def correlation_test(self, df: pd.DataFrame, col1: str, col2: str) -> Dict:
        """
        Pearson correlation with significance test
        """
        # Remove NaN values
        clean_df = df[[col1, col2]].dropna()
        
        if len(clean_df) < 3:
            return {'error': 'Insufficient data for correlation test'}
        
        # Pearson correlation
        r, p_value = pearsonr(clean_df[col1], clean_df[col2])
        
        return {
            'test': 'Pearson Correlation',
            'correlation_coefficient': r,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'strength': self._interpret_correlation(r),
            'direction': 'positive' if r > 0 else 'negative'
        }
    
    def t_test(self, df: pd.DataFrame, group_col: str, value_col: str) -> Dict:
        """
        Independent samples t-test (for two groups)
        """
        groups = df.groupby(group_col)[value_col].apply(list)
        
        if len(groups) != 2:
            return {'error': 'T-test requires exactly 2 groups'}
        
        group1, group2 = groups.iloc[0], groups.iloc[1]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Effect size (Cohen's d)
        cohens_d = (np.mean(group1) - np.mean(group2)) / np.sqrt(
            ((len(group1) - 1) * np.var(group1) + (len(group2) - 1) * np.var(group2)) / 
            (len(group1) + len(group2) - 2)
        )
        
        return {
            'test': 'Independent T-Test',
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'effect_size_cohens_d': cohens_d,
            'interpretation': self._interpret_cohens_d(cohens_d),
            'group_means': {
                str(groups.index[0]): np.mean(group1),
                str(groups.index[1]): np.mean(group2)
            }
        }
    
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V effect size"""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta: float) -> str:
        """Interpret eta-squared effect size"""
        if eta < 0.01:
            return "negligible"
        elif eta < 0.06:
            return "small"
        elif eta < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation strength"""
        abs_r = abs(r)
        if abs_r < 0.3:
            return "weak"
        elif abs_r < 0.7:
            return "moderate"
        else:
            return "strong"
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"


class InteractionAnalyzer:
    """
    Analyze interaction effects between multiple variables
    """
    
    def __init__(self):
        self.results = {}
    
    def two_way_interaction(self, df: pd.DataFrame, 
                           factor1: str, factor2: str, 
                           response: str) -> Dict:
        """
        Analyze two-way interaction effect
        """
        # Create interaction groups
        interaction_pivot = df.pivot_table(
            index=factor1,
            columns=factor2,
            values=response,
            aggfunc='mean'
        )
        
        # Calculate interaction strength
        # If lines are parallel in interaction plot, no interaction
        # If lines cross, strong interaction
        
        results = {
            'pivot_table': interaction_pivot.to_dict(),
            'factor1_main_effect': df.groupby(factor1)[response].mean().to_dict(),
            'factor2_main_effect': df.groupby(factor2)[response].mean().to_dict()
        }
        
        # Interaction strength (variance of cell means around marginal means)
        if not interaction_pivot.empty:
            grand_mean = df[response].mean()
            
            # Sum of squares for interaction
            ss_interaction = 0
            for i in interaction_pivot.index:
                for j in interaction_pivot.columns:
                    cell_mean = interaction_pivot.loc[i, j]
                    row_mean = df[df[factor1] == i][response].mean()
                    col_mean = df[df[factor2] == j][response].mean()
                    
                    if not pd.isna(cell_mean):
                        expected = row_mean + col_mean - grand_mean
                        ss_interaction += (cell_mean - expected) ** 2
            
            results['interaction_strength'] = ss_interaction
            results['has_interaction'] = ss_interaction > (df[response].var() * 0.1)
        
        return results
    
    def three_way_interaction(self, df: pd.DataFrame,
                             factor1: str, factor2: str, factor3: str,
                             response: str) -> Dict:
        """
        Analyze three-way interaction
        """
        # Three-way aggregation
        three_way = df.groupby([factor1, factor2, factor3])[response].agg(['mean', 'std', 'count'])
        
        results = {
            'three_way_summary': three_way.to_dict(),
            'sample_sizes': three_way['count'].to_dict()
        }
        
        # Check if all combinations have sufficient data
        min_count = three_way['count'].min()
        results['sufficient_data'] = min_count >= 5
        results['min_sample_size'] = min_count
        
        return results
    
    def analyze_all_pairs(self, df: pd.DataFrame, 
                         categorical_cols: List[str],
                         response_col: str) -> Dict:
        """
        Analyze all pairwise interactions
        """
        results = {}
        
        for col1, col2 in combinations(categorical_cols, 2):
            key = f"{col1}_x_{col2}"
            results[key] = self.two_way_interaction(df, col1, col2, response_col)
        
        return results


def run_comprehensive_statistical_analysis(df: pd.DataFrame,
                                          categorical_cols: List[str],
                                          numeric_cols: List[str]) -> Dict:
    """
    Run comprehensive statistical analysis on dataset
    """
    tester = SignificanceTest()
    results = {}
    
    # Chi-square tests for categorical pairs
    results['chi_square_tests'] = {}
    for col1, col2 in combinations(categorical_cols, 2):
        if col1 in df.columns and col2 in df.columns:
            try:
                results['chi_square_tests'][f'{col1}_vs_{col2}'] = tester.chi_square_test(df, col1, col2)
            except Exception as e:
                results['chi_square_tests'][f'{col1}_vs_{col2}'] = {'error': str(e)}
    
    # ANOVA tests
    results['anova_tests'] = {}
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            if cat_col in df.columns and num_col in df.columns:
                try:
                    results['anova_tests'][f'{num_col}_by_{cat_col}'] = tester.anova_test(df, cat_col, num_col)
                except Exception as e:
                    results['anova_tests'][f'{num_col}_by_{cat_col}'] = {'error': str(e)}
    
    # Correlation tests
    results['correlation_tests'] = {}
    for col1, col2 in combinations(numeric_cols, 2):
        if col1 in df.columns and col2 in df.columns:
            try:
                results['correlation_tests'][f'{col1}_vs_{col2}'] = tester.correlation_test(df, col1, col2)
            except Exception as e:
                results['correlation_tests'][f'{col1}_vs_{col2}'] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    print("Advanced Statistics Module")
    print("=" * 60)
    print("Available Analyses:")
    print("  1. Trivariate Analysis (Age × Geography × Time)")
    print("  2. Chi-Square Tests (Categorical Independence)")
    print("  3. ANOVA Tests (Group Differences)")
    print("  4. Correlation Tests (Relationship Strength)")
    print("  5. Interaction Effects (Two-way & Three-way)")
    print("=" * 60)
