"""
Visualization Module for UIDAI Data Hackathon 2026
Reusable visualization functions for analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class AadhaarVisualizer:
    """Visualization utilities for Aadhaar data analysis"""
    
    def __init__(self, output_dir: str = '../outputs/figures'):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        self.color_palette = sns.color_palette('husl', 10)
        
    def plot_temporal_trend(
        self, 
        df: pd.DataFrame, 
        value_col: str,
        date_col: str = 'date',
        title: str = 'Temporal Trend',
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot time series trend
        
        Args:
            df: DataFrame with date and value columns
            value_col: Column to plot
            date_col: Date column name
            title: Plot title
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Aggregate by date if needed
        df_agg = df.groupby(date_col)[value_col].sum().reset_index()
        
        ax.plot(df_agg[date_col], df_agg[value_col], linewidth=2, color='#2E86AB')
        ax.fill_between(df_agg[date_col], df_agg[value_col], alpha=0.3, color='#2E86AB')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.output_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_geographic_distribution(
        self,
        df: pd.DataFrame,
        geo_col: str,
        value_col: str,
        top_n: int = 20,
        title: str = 'Geographic Distribution',
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot geographic distribution
        
        Args:
            df: DataFrame
            geo_col: Geographic column ('state', 'district', etc.)
            value_col: Value to plot
            top_n: Number of top locations to show
            title: Plot title
            save_name: Filename to save
        """
        # Aggregate by geography
        df_agg = df.groupby(geo_col)[value_col].sum().sort_values(ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        df_agg.plot(kind='barh', ax=ax, color='#A23B72')
        
        ax.set_xlabel(value_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(geo_col.title(), fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(df_agg.values):
            ax.text(v, i, f' {v:,.0f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.output_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_age_distribution(
        self,
        df: pd.DataFrame,
        age_cols: List[str],
        labels: List[str],
        title: str = 'Age Group Distribution',
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot age group distribution
        
        Args:
            df: DataFrame
            age_cols: List of age group columns
            labels: Labels for age groups
            title: Plot title
            save_name: Filename to save
        """
        # Sum across all records
        totals = [df[col].sum() for col in age_cols]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        colors = sns.color_palette('Set2', len(age_cols))
        ax1.pie(totals, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.set_title('Proportion by Age Group', fontweight='bold')
        
        # Bar chart
        ax2.bar(labels, totals, color=colors)
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Total by Age Group', fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
        
        # Add value labels
        for i, v in enumerate(totals):
            ax2.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.output_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_heatmap(
        self,
        df: pd.DataFrame,
        index_col: str,
        columns_col: str,
        value_col: str,
        title: str = 'Heatmap',
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot heatmap
        
        Args:
            df: DataFrame
            index_col: Column for y-axis
            columns_col: Column for x-axis
            value_col: Value to plot
            title: Plot title
            save_name: Filename to save
        """
        # Create pivot table
        pivot_df = df.pivot_table(
            index=index_col,
            columns=columns_col,
            values=value_col,
            aggfunc='sum'
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            pivot_df,
            cmap='YlOrRd',
            annot=False,
            fmt='.0f',
            cbar_kws={'label': value_col.replace('_', ' ').title()},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(columns_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(index_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.output_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = 'Correlation Matrix',
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot correlation matrix
        
        Args:
            df: DataFrame
            columns: Columns to include (None = all numeric)
            title: Plot title
            save_name: Filename to save
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = df[columns].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.output_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_seasonal_decomposition(
        self,
        df: pd.DataFrame,
        value_col: str,
        date_col: str = 'date',
        freq: int = 7,
        title: str = 'Seasonal Decomposition',
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot seasonal decomposition
        
        Args:
            df: DataFrame
            value_col: Column to decompose
            date_col: Date column
            freq: Seasonal frequency
            title: Plot title
            save_name: Filename to save
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Aggregate by date
        df_agg = df.groupby(date_col)[value_col].sum().reset_index()
        df_agg = df_agg.set_index(date_col)
        
        # Perform decomposition
        decomposition = seasonal_decompose(df_agg[value_col], model='additive', period=freq)
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        decomposition.observed.plot(ax=axes[0], color='#2E86AB')
        axes[0].set_ylabel('Observed', fontweight='bold')
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        
        decomposition.trend.plot(ax=axes[1], color='#A23B72')
        axes[1].set_ylabel('Trend', fontweight='bold')
        
        decomposition.seasonal.plot(ax=axes[2], color='#F18F01')
        axes[2].set_ylabel('Seasonal', fontweight='bold')
        
        decomposition.resid.plot(ax=axes[3], color='#C73E1D')
        axes[3].set_ylabel('Residual', fontweight='bold')
        axes[3].set_xlabel('Date', fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.output_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_choropleth(
        self,
        df: pd.DataFrame,
        geo_col: str,
        value_col: str,
        title: str = 'Geographic Distribution',
        save_name: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive choropleth map (for states)
        
        Args:
            df: DataFrame
            geo_col: Geographic column
            value_col: Value to plot
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Plotly figure object
        """
        df_agg = df.groupby(geo_col)[value_col].sum().reset_index()
        
        fig = px.choropleth(
            df_agg,
            locations=geo_col,
            locationmode='country names',
            color=value_col,
            hover_name=geo_col,
            color_continuous_scale='Viridis',
            title=title
        )
        
        fig.update_layout(
            geo=dict(
                scope='asia',
                center=dict(lat=23, lon=80),
                projection_scale=4
            ),
            height=600
        )
        
        if save_name:
            fig.write_html(f"{self.output_dir}/{save_name}")
        
        return fig
    
    def plot_box_comparison(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        title: str = 'Distribution Comparison',
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot box plot comparison
        
        Args:
            df: DataFrame
            group_col: Grouping column
            value_col: Value column
            title: Plot title
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Get top groups by total value
        top_groups = df.groupby(group_col)[value_col].sum().nlargest(15).index
        df_filtered = df[df[group_col].isin(top_groups)]
        
        sns.boxplot(
            data=df_filtered,
            x=group_col,
            y=value_col,
            palette='Set2',
            ax=ax
        )
        
        ax.set_xlabel(group_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.output_dir}/{save_name}", dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("Example usage:")
    print("  visualizer = AadhaarVisualizer()")
    print("  visualizer.plot_temporal_trend(df, 'total_enrolments')")
