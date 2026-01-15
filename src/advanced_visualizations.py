"""
Advanced Visualization Module
Creates innovative and interactive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class AdvancedVisualizer:
    """
    Creates advanced, creative visualizations beyond standard charts
    """
    
    def __init__(self, output_dir='outputs/figures'):
        self.output_dir = output_dir
        self.color_palettes = {
            'primary': ['#3A86FF', '#FB5607', '#8338EC', '#06D6A0', '#FFD60A'],
            'sequential': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6'],
            'diverging': ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']
        }
    
    def create_3d_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, 
                         z_col: str, color_col: str = None, 
                         title: str = "3D Scatter Plot") -> go.Figure:
        """
        Create 3D scatter plot for trivariate analysis
        """
        fig = px.scatter_3d(
            df, x=x_col, y=y_col, z=z_col,
            color=color_col if color_col else None,
            title=title,
            labels={x_col: x_col.replace('_', ' ').title(),
                   y_col: y_col.replace('_', ' ').title(),
                   z_col: z_col.replace('_', ' ').title()},
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                zaxis_title=z_col.replace('_', ' ').title()
            ),
            font=dict(size=12),
            height=700
        )
        
        return fig
    
    def create_sankey_diagram(self, df: pd.DataFrame, source_col: str,
                             target_col: str, value_col: str = None,
                             title: str = "Flow Diagram") -> go.Figure:
        """
        Create Sankey diagram for flow visualization
        """
        # Prepare data
        if value_col:
            flow_data = df.groupby([source_col, target_col])[value_col].sum().reset_index()
        else:
            flow_data = df.groupby([source_col, target_col]).size().reset_index(name='value')
        
        # Create node labels
        sources = flow_data[source_col].unique()
        targets = flow_data[target_col].unique()
        all_nodes = list(sources) + list(targets)
        all_nodes = list(dict.fromkeys(all_nodes))  # Remove duplicates
        
        # Map to indices
        node_dict = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Create links
        source_indices = [node_dict[s] for s in flow_data[source_col]]
        target_indices = [node_dict[t] for t in flow_data[target_col]]
        values = flow_data[value_col if value_col else 'value'].tolist()
        
        # Create figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color="blue"
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values
            )
        )])
        
        fig.update_layout(title_text=title, font_size=12, height=600)
        
        return fig
    
    def create_ridge_plot(self, df: pd.DataFrame, group_col: str, 
                         value_col: str, title: str = "Ridge Plot") -> plt.Figure:
        """
        Create ridge plot (joyplot) for distribution comparison
        """
        groups = sorted(df[group_col].unique())
        
        fig, axes = plt.subplots(len(groups), 1, figsize=(12, len(groups) * 1.5),
                                sharex=True)
        
        if len(groups) == 1:
            axes = [axes]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
        
        for idx, (group, ax) in enumerate(zip(groups, axes)):
            group_data = df[df[group_col] == group][value_col].dropna()
            
            if len(group_data) > 0:
                ax.fill_between(
                    np.linspace(group_data.min(), group_data.max(), 100),
                    0,
                    np.histogram(group_data, bins=100, density=True)[0],
                    alpha=0.7,
                    color=colors[idx]
                )
                
                ax.set_ylabel(str(group), rotation=0, labelpad=50, fontsize=10)
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                if idx < len(groups) - 1:
                    ax.spines['bottom'].set_visible(False)
                    ax.set_xticks([])
        
        axes[-1].set_xlabel(value_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        return fig
    
    def create_violin_comparison(self, df: pd.DataFrame, group_col: str,
                                value_col: str, title: str = "Violin Plot") -> plt.Figure:
        """
        Create violin plot with box plot overlay
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Violin plot
        parts = ax.violinplot(
            [df[df[group_col] == g][value_col].dropna().values 
             for g in sorted(df[group_col].unique())],
            positions=range(len(df[group_col].unique())),
            showmeans=True,
            showmedians=True
        )
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('#3A86FF')
            pc.set_alpha(0.7)
        
        # Box plot overlay
        bp = ax.boxplot(
            [df[df[group_col] == g][value_col].dropna().values 
             for g in sorted(df[group_col].unique())],
            positions=range(len(df[group_col].unique())),
            widths=0.15,
            patch_artist=True,
            showfliers=False
        )
        
        for patch in bp['boxes']:
            patch.set_facecolor('#FB5607')
            patch.set_alpha(0.6)
        
        ax.set_xticks(range(len(df[group_col].unique())))
        ax.set_xticklabels(sorted(df[group_col].unique()), rotation=45, ha='right')
        ax.set_xlabel(group_col.replace('_', ' ').title(), fontweight='bold', fontsize=12)
        ax.set_ylabel(value_col.replace('_', ' ').title(), fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_network_graph(self, df: pd.DataFrame, source_col: str,
                            target_col: str, title: str = "Network Graph") -> go.Figure:
        """
        Create network graph for relationship visualization
        """
        # Aggregate edges
        edges = df.groupby([source_col, target_col]).size().reset_index(name='weight')
        
        # Get unique nodes
        nodes = list(set(list(edges[source_col].unique()) + list(edges[target_col].unique())))
        node_dict = {node: idx for idx, node in enumerate(nodes)}
        
        # Create edge traces
        edge_traces = []
        for _, row in edges.iterrows():
            source_idx = node_dict[row[source_col]]
            target_idx = node_dict[row[target_col]]
            
            # Simple layout (circular)
            angle_source = 2 * np.pi * source_idx / len(nodes)
            angle_target = 2 * np.pi * target_idx / len(nodes)
            
            x_source, y_source = np.cos(angle_source), np.sin(angle_source)
            x_target, y_target = np.cos(angle_target), np.sin(angle_target)
            
            edge_trace = go.Scatter(
                x=[x_source, x_target, None],
                y=[y_source, y_target, None],
                mode='lines',
                line=dict(width=row['weight'] / edges['weight'].max() * 5, color='gray'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = [np.cos(2 * np.pi * i / len(nodes)) for i in range(len(nodes))]
        node_y = [np.sin(2 * np.pi * i / len(nodes)) for i in range(len(nodes))]
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=nodes,
            textposition='top center',
            marker=dict(size=20, color='#3A86FF', line=dict(width=2, color='white')),
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700
        )
        
        return fig
    
    def create_heatmap_with_dendrogram(self, df: pd.DataFrame, 
                                      title: str = "Clustered Heatmap") -> plt.Figure:
        """
        Create heatmap with hierarchical clustering
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        
        # Prepare data (numeric only)
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for heatmap")
        
        # Calculate linkage
        row_linkage = linkage(pdist(numeric_df, metric='euclidean'), method='ward')
        col_linkage = linkage(pdist(numeric_df.T, metric='euclidean'), method='ward')
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        
        # Dendrogram for rows
        ax1 = fig.add_axes([0.05, 0.1, 0.15, 0.6])
        dendrogram(row_linkage, orientation='left', ax=ax1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Dendrogram for columns
        ax2 = fig.add_axes([0.25, 0.75, 0.6, 0.15])
        dendrogram(col_linkage, ax=ax2)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Heatmap
        ax3 = fig.add_axes([0.25, 0.1, 0.6, 0.6])
        
        # Reorder based on dendrograms
        row_order = dendrogram(row_linkage, no_plot=True)['leaves']
        col_order = dendrogram(col_linkage, no_plot=True)['leaves']
        
        ordered_data = numeric_df.iloc[row_order, col_order]
        
        im = ax3.imshow(ordered_data, aspect='auto', cmap='RdYlBu_r')
        ax3.set_xticks(range(len(ordered_data.columns)))
        ax3.set_xticklabels(ordered_data.columns, rotation=90)
        ax3.set_yticks(range(len(ordered_data.index)))
        ax3.set_yticklabels(ordered_data.index)
        
        # Colorbar
        cax = fig.add_axes([0.87, 0.1, 0.02, 0.6])
        plt.colorbar(im, cax=cax)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        return fig
    
    def create_animated_time_series(self, df: pd.DataFrame, date_col: str,
                                   value_col: str, group_col: str = None,
                                   title: str = "Animated Time Series") -> go.Figure:
        """
        Create animated time series visualization
        """
        if group_col:
            fig = px.line(
                df, x=date_col, y=value_col, color=group_col,
                animation_frame=df[date_col].dt.year if hasattr(df[date_col], 'dt') else None,
                title=title,
                labels={value_col: value_col.replace('_', ' ').title()}
            )
        else:
            fig = px.line(
                df, x=date_col, y=value_col,
                title=title,
                labels={value_col: value_col.replace('_', ' ').title()}
            )
        
        fig.update_layout(
            xaxis_title=date_col.replace('_', ' ').title(),
            yaxis_title=value_col.replace('_', ' ').title(),
            font=dict(size=12),
            height=600
        )
        
        return fig
    
    def create_sunburst_chart(self, df: pd.DataFrame, path_cols: List[str],
                             value_col: str, title: str = "Sunburst Chart") -> go.Figure:
        """
        Create sunburst chart for hierarchical data
        """
        fig = px.sunburst(
            df, path=path_cols, values=value_col,
            title=title,
            color=value_col,
            color_continuous_scale='RdYlBu_r'
        )
        
        fig.update_layout(font=dict(size=12), height=700)
        
        return fig
    
    def create_parallel_coordinates(self, df: pd.DataFrame, 
                                   numeric_cols: List[str],
                                   color_col: str = None,
                                   title: str = "Parallel Coordinates") -> go.Figure:
        """
        Create parallel coordinates plot for multi-dimensional data
        """
        # Normalize data
        normalized_df = df[numeric_cols].copy()
        for col in numeric_cols:
            normalized_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        if color_col and color_col in df.columns:
            normalized_df[color_col] = df[color_col]
        
        fig = px.parallel_coordinates(
            normalized_df,
            dimensions=numeric_cols,
            color=color_col if color_col else numeric_cols[0],
            title=title,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(font=dict(size=11), height=600)
        
        return fig
    
    def save_figure(self, fig, filename: str, format: str = 'png'):
        """
        Save figure to file (supports matplotlib and plotly)
        """
        filepath = f"{self.output_dir}/{filename}"
        
        if isinstance(fig, plt.Figure):
            fig.savefig(f"{filepath}.{format}", dpi=300, bbox_inches='tight')
        elif isinstance(fig, go.Figure):
            if format == 'html':
                fig.write_html(f"{filepath}.html")
            else:
                fig.write_image(f"{filepath}.{format}", width=1400, height=800)
        
        print(f"✓ Saved: {filepath}.{format}")


if __name__ == "__main__":
    print("Advanced Visualization Module")
    print("=" * 60)
    print("Available Visualizations:")
    print("  1. 3D Scatter Plot")
    print("  2. Sankey Diagram")
    print("  3. Ridge Plot (Joyplot)")
    print("  4. Violin Plot with Box Plot Overlay")
    print("  5. Network Graph")
    print("  6. Clustered Heatmap with Dendrogram")
    print("  7. Animated Time Series")
    print("  8. Sunburst Chart")
    print("  9. Parallel Coordinates")
    print("=" * 60)
