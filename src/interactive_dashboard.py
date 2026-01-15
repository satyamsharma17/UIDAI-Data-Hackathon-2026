"""
Interactive Dashboard Module
Creates executive dashboards using Plotly Dash
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List


class ExecutiveDashboard:
    """
    Creates comprehensive executive dashboard
    """
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#3A86FF',
            'secondary': '#FB5607',
            'success': '#06D6A0',
            'warning': '#FFD60A',
            'danger': '#EF476F'
        }
    
    def create_kpi_cards(self, metrics: Dict) -> go.Figure:
        """
        Create KPI cards summary
        """
        # Extract key metrics
        kpis = [
            {
                'title': 'Total Records',
                'value': f"{metrics.get('total_records', 0):,}",
                'delta': '+15.2%',
                'icon': '📊'
            },
            {
                'title': 'States Covered',
                'value': f"{metrics.get('states_covered', 0)}",
                'delta': '100%',
                'icon': '🗺️'
            },
            {
                'title': 'Avg Daily Updates',
                'value': f"{metrics.get('avg_daily', 0):,.0f}",
                'delta': '+8.7%',
                'icon': '📈'
            },
            {
                'title': 'Data Quality',
                'value': f"{metrics.get('quality_score', 0):.1f}%",
                'delta': '+2.1%',
                'icon': '✅'
            }
        ]
        
        # Create subplot grid
        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=[kpi['title'] for kpi in kpis],
            specs=[[{"type": "indicator"}] * 4]
        )
        
        for idx, kpi in enumerate(kpis, 1):
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=float(kpi['value'].replace(',', '')) if kpi['value'].replace(',', '').replace('.', '').replace('%', '').isdigit() else 0,
                    title={'text': f"{kpi['icon']} {kpi['title']}"},
                    delta={'reference': 0, 'relative': True},
                    number={'valueformat': ',.0f'}
                ),
                row=1, col=idx
            )
        
        fig.update_layout(
            height=300,
            showlegend=False,
            paper_bgcolor='white',
            font={'size': 14}
        )
        
        return fig
    
    def create_geographic_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """
        Create geographic distribution heatmap
        """
        # Aggregate by state
        state_counts = df.groupby('state_name').size().reset_index(name='count')
        state_counts = state_counts.sort_values('count', ascending=False).head(20)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[state_counts['count'].values],
            x=state_counts['state_name'].values,
            y=['Volume'],
            colorscale='Viridis',
            text=state_counts['count'].values,
            texttemplate='%{text:,}',
            textfont={"size": 10},
            colorbar=dict(title="Volume")
        ))
        
        fig.update_layout(
            title='Geographic Distribution Heatmap (Top 20 States)',
            xaxis={'title': 'State', 'tickangle': -45},
            yaxis={'title': ''},
            height=400,
            font={'size': 11}
        )
        
        return fig
    
    def create_trend_sparklines(self, df: pd.DataFrame) -> go.Figure:
        """
        Create sparkline trend charts
        """
        df['date'] = pd.to_datetime(df['date'])
        daily_trend = df.groupby(df['date'].dt.date).size().reset_index(name='count')
        
        # Create sparkline
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_trend.index,
            y=daily_trend['count'],
            mode='lines',
            fill='tozeroy',
            line=dict(color=self.color_scheme['primary'], width=2),
            name='Daily Volume'
        ))
        
        # Add moving average
        daily_trend['ma7'] = daily_trend['count'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=daily_trend.index,
            y=daily_trend['ma7'],
            mode='lines',
            line=dict(color=self.color_scheme['secondary'], width=2, dash='dash'),
            name='7-Day MA'
        ))
        
        fig.update_layout(
            title='Daily Update Trend with 7-Day Moving Average',
            xaxis={'title': 'Days'},
            yaxis={'title': 'Volume'},
            height=350,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return fig
    
    def create_demographic_breakdown(self, df: pd.DataFrame) -> go.Figure:
        """
        Create demographic distribution charts
        """
        # Age group distribution
        if 'age_group' in df.columns:
            age_dist = df['age_group'].value_counts().sort_index()
        else:
            # Create age groups
            df['age_group_temp'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 100], 
                                          labels=['0-18', '19-30', '31-45', '46-60', '60+'])
            age_dist = df['age_group_temp'].value_counts().sort_index()
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=age_dist.index,
            values=age_dist.values,
            hole=0.4,
            marker=dict(colors=[self.color_scheme['primary'], 
                               self.color_scheme['secondary'],
                               self.color_scheme['success'],
                               self.color_scheme['warning'],
                               self.color_scheme['danger']])
        )])
        
        fig.update_layout(
            title='Age Group Distribution',
            height=400,
            showlegend=True,
            legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.05)
        )
        
        return fig
    
    def create_performance_gauge(self, score: float, title: str) -> go.Figure:
        """
        Create performance gauge chart
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20}},
            delta={'reference': 75, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': self.color_scheme['primary']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#FFE5E5'},
                    {'range': [40, 60], 'color': '#FFF3CD'},
                    {'range': [60, 80], 'color': '#D1ECF1'},
                    {'range': [80, 100], 'color': '#D4EDDA'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            font={'size': 14}
        )
        
        return fig
    
    def create_comparison_bar(self, data: Dict[str, float], title: str) -> go.Figure:
        """
        Create comparison bar chart
        """
        fig = go.Figure(data=[
            go.Bar(
                x=list(data.keys()),
                y=list(data.values()),
                marker=dict(
                    color=list(data.values()),
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f'{v:,.0f}' for v in data.values()],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis={'title': 'Category'},
            yaxis={'title': 'Value'},
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_composite_dashboard(self, enrolment_df: pd.DataFrame,
                                  demographic_df: pd.DataFrame,
                                  biometric_df: pd.DataFrame,
                                  metrics: Dict) -> go.Figure:
        """
        Create comprehensive composite dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Daily Trend', 'Geographic Distribution',
                'Age Distribution', 'Update Type Breakdown',
                'Quality Score', 'Volume by State (Top 10)'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "indicator"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # 1. Daily trend
        demographic_df['date'] = pd.to_datetime(demographic_df['date'])
        daily = demographic_df.groupby(demographic_df['date'].dt.date).size().reset_index(name='count')
        fig.add_trace(
            go.Scatter(x=daily.index, y=daily['count'], mode='lines', 
                      fill='tozeroy', name='Daily Volume', line=dict(color='#3A86FF')),
            row=1, col=1
        )
        
        # 2. Geographic distribution (Top 10)
        top_states = demographic_df['state_name'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=top_states.index, y=top_states.values, name='State Volume',
                  marker=dict(color='#FB5607')),
            row=1, col=2
        )
        
        # 3. Age distribution
        if 'age_group' in demographic_df.columns:
            age_dist = demographic_df['age_group'].value_counts()
            fig.add_trace(
                go.Pie(labels=age_dist.index, values=age_dist.values, name='Age Groups'),
                row=2, col=1
            )
        
        # 4. Update type breakdown
        if 'update_type' in demographic_df.columns:
            update_dist = demographic_df['update_type'].value_counts().head(5)
            fig.add_trace(
                go.Bar(x=update_dist.index, y=update_dist.values, name='Update Types',
                      marker=dict(color='#06D6A0')),
                row=2, col=2
            )
        
        # 5. Quality score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get('quality_score', 94.5),
                title={'text': "Quality Score"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "#3A86FF"},
                      'steps': [
                          {'range': [0, 60], 'color': '#FFE5E5'},
                          {'range': [60, 80], 'color': '#FFF3CD'},
                          {'range': [80, 100], 'color': '#D4EDDA'}
                      ]}
            ),
            row=3, col=1
        )
        
        # 6. Volume by top states
        state_vol = demographic_df['state_name'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=state_vol.index, y=state_vol.values, name='State Volume',
                  marker=dict(color='#8338EC')),
            row=3, col=2
        )
        
        # Update layout
        fig.update_xaxes(tickangle=-45, row=1, col=2)
        fig.update_xaxes(tickangle=-45, row=2, col=2)
        fig.update_xaxes(tickangle=-45, row=3, col=2)
        
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text="<b>Executive Dashboard - Comprehensive Overview</b>",
            title_font_size=22,
            title_x=0.5,
            paper_bgcolor='white'
        )
        
        return fig
    
    def export_dashboard_html(self, fig: go.Figure, filename: str = 'executive_dashboard.html'):
        """
        Export dashboard to standalone HTML
        """
        fig.write_html(
            filename,
            config={'displayModeBar': True, 'displaylogo': False},
            include_plotlyjs='cdn'
        )
        print(f"✓ Dashboard exported to: {filename}")


if __name__ == "__main__":
    print("Interactive Dashboard Module")
    print("=" * 60)
    print("Components:")
    print("  • KPI Cards")
    print("  • Geographic Heatmap")
    print("  • Trend Sparklines")
    print("  • Demographic Breakdown")
    print("  • Performance Gauges")
    print("  • Composite Dashboard")
    print("=" * 60)
