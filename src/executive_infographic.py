"""
Executive Infographic Generator for UIDAI Data Hackathon 2026
============================================================

This module creates one-page executive infographics that summarize
key findings, insights, and recommendations in a visually compelling format.

Author: UIDAI Hackathon Team
Date: 2026
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutiveInfographic:
    """
    Generates one-page executive infographics summarizing analysis findings.
    
    This class creates visually compelling single-page summaries that combine
    data visualizations, key metrics, and strategic insights in an easy-to-digest format.
    """
    
    def __init__(self, output_dir: str = "infographics"):
        """
        Initialize the Executive Infographic generator.
        
        Args:
            output_dir: Directory to save infographic files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color palette
        self.colors = {
            'primary': '#0066CC',
            'secondary': '#FF6600',
            'success': '#00CC66',
            'warning': '#FFB800',
            'danger': '#CC0000',
            'neutral': '#666666',
            'light': '#F5F5F5',
            'dark': '#333333'
        }
        
        # Typography
        self.fonts = {
            'title': {'family': 'sans-serif', 'weight': 'bold', 'size': 24},
            'heading': {'family': 'sans-serif', 'weight': 'bold', 'size': 16},
            'subheading': {'family': 'sans-serif', 'weight': 'bold', 'size': 12},
            'body': {'family': 'sans-serif', 'weight': 'normal', 'size': 10},
            'metric': {'family': 'sans-serif', 'weight': 'bold', 'size': 20}
        }
        
        logger.info(f"ExecutiveInfographic initialized. Output: {self.output_dir}")
    
    def create_header(self, ax: plt.Axes, title: str, subtitle: str) -> None:
        """
        Create infographic header with title and subtitle.
        
        Args:
            ax: Matplotlib axes
            title: Main title
            subtitle: Subtitle text
        """
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.7, title, 
               ha='center', va='center',
               fontsize=self.fonts['title']['size'],
               fontweight=self.fonts['title']['weight'],
               color=self.colors['primary'])
        
        # Subtitle
        ax.text(0.5, 0.3, subtitle,
               ha='center', va='center',
               fontsize=self.fonts['heading']['size'],
               color=self.colors['neutral'])
        
        # Date
        date_str = datetime.now().strftime("%B %Y")
        ax.text(0.95, 0.1, date_str,
               ha='right', va='center',
               fontsize=self.fonts['body']['size'],
               color=self.colors['neutral'],
               style='italic')
    
    def create_metric_card(self, ax: plt.Axes, value: str, label: str, 
                          icon: str = "", color: str = 'primary') -> None:
        """
        Create metric display card.
        
        Args:
            ax: Matplotlib axes
            value: Metric value
            label: Metric label
            icon: Optional icon/emoji
            color: Color scheme key
        """
        ax.axis('off')
        
        # Background
        rect = mpatches.FancyBboxPatch(
            (0.05, 0.1), 0.9, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=self.colors['light'],
            edgecolor=self.colors[color],
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Icon
        if icon:
            ax.text(0.5, 0.75, icon,
                   ha='center', va='center',
                   fontsize=24)
        
        # Value
        ax.text(0.5, 0.5, value,
               ha='center', va='center',
               fontsize=self.fonts['metric']['size'],
               fontweight=self.fonts['metric']['weight'],
               color=self.colors[color])
        
        # Label
        ax.text(0.5, 0.25, label,
               ha='center', va='center',
               fontsize=self.fonts['subheading']['size'],
               color=self.colors['neutral'])
    
    def create_mini_bar_chart(self, ax: plt.Axes, data: Dict[str, float],
                             title: str, color: str = 'primary') -> None:
        """
        Create compact bar chart.
        
        Args:
            ax: Matplotlib axes
            data: Dictionary of labels and values
            title: Chart title
            color: Color scheme key
        """
        labels = list(data.keys())
        values = list(data.values())
        
        ax.barh(labels, values, color=self.colors[color], alpha=0.7)
        ax.set_title(title, fontsize=self.fonts['subheading']['size'],
                    fontweight=self.fonts['subheading']['weight'],
                    pad=10)
        ax.set_xlabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=self.fonts['body']['size'])
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v, i, f' {v:.1f}', 
                   va='center', fontsize=self.fonts['body']['size'])
    
    def create_mini_pie_chart(self, ax: plt.Axes, data: Dict[str, float],
                             title: str) -> None:
        """
        Create compact pie chart.
        
        Args:
            ax: Matplotlib axes
            data: Dictionary of labels and values
            title: Chart title
        """
        labels = list(data.keys())
        values = list(data.values())
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['success'], self.colors['warning']]
        
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=colors[:len(values)],
            startangle=90,
            textprops={'fontsize': self.fonts['body']['size']}
        )
        
        # Bold percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=self.fonts['subheading']['size'],
                    fontweight=self.fonts['subheading']['weight'],
                    pad=10)
    
    def create_timeline_visual(self, ax: plt.Axes, milestones: List[Dict[str, str]]) -> None:
        """
        Create visual timeline.
        
        Args:
            ax: Matplotlib axes
            milestones: List of milestone dictionaries
        """
        ax.axis('off')
        
        n_milestones = len(milestones)
        positions = np.linspace(0.1, 0.9, n_milestones)
        
        # Timeline line
        ax.plot([0.1, 0.9], [0.5, 0.5], 
               color=self.colors['primary'], linewidth=3)
        
        # Milestones
        for i, (pos, milestone) in enumerate(zip(positions, milestones)):
            # Dot
            ax.plot(pos, 0.5, 'o', 
                   markersize=15, 
                   color=self.colors['primary'],
                   markeredgecolor='white',
                   markeredgewidth=2)
            
            # Label (alternating above/below)
            y_label = 0.7 if i % 2 == 0 else 0.3
            ax.text(pos, y_label, milestone['label'],
                   ha='center', va='center',
                   fontsize=self.fonts['body']['size'],
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='white',
                           edgecolor=self.colors['primary']))
    
    def create_key_insights_panel(self, ax: plt.Axes, insights: List[str]) -> None:
        """
        Create key insights panel.
        
        Args:
            ax: Matplotlib axes
            insights: List of key insight strings
        """
        ax.axis('off')
        
        # Title
        ax.text(0.05, 0.95, "🔑 Key Insights",
               fontsize=self.fonts['heading']['size'],
               fontweight=self.fonts['heading']['weight'],
               color=self.colors['primary'])
        
        # Insights
        y_pos = 0.85
        for i, insight in enumerate(insights[:5]):  # Max 5 insights
            # Bullet point
            ax.text(0.05, y_pos, f"{i+1}.",
                   fontsize=self.fonts['body']['size'],
                   fontweight='bold',
                   color=self.colors['primary'])
            
            # Insight text
            ax.text(0.12, y_pos, insight,
                   fontsize=self.fonts['body']['size'],
                   color=self.colors['dark'],
                   wrap=True,
                   verticalalignment='top')
            
            y_pos -= 0.15
    
    def create_recommendations_panel(self, ax: plt.Axes, 
                                    recommendations: List[Dict[str, str]]) -> None:
        """
        Create recommendations panel.
        
        Args:
            ax: Matplotlib axes
            recommendations: List of recommendation dictionaries
        """
        ax.axis('off')
        
        # Title
        ax.text(0.05, 0.95, "🎯 Top Recommendations",
               fontsize=self.fonts['heading']['size'],
               fontweight=self.fonts['heading']['weight'],
               color=self.colors['secondary'])
        
        # Recommendations
        y_pos = 0.85
        priority_colors = {
            'high': self.colors['danger'],
            'medium': self.colors['warning'],
            'low': self.colors['success']
        }
        
        for rec in recommendations[:5]:  # Max 5 recommendations
            priority = rec.get('priority', 'medium').lower()
            color = priority_colors.get(priority, self.colors['neutral'])
            
            # Priority indicator
            ax.text(0.05, y_pos, "●",
                   fontsize=16,
                   color=color)
            
            # Recommendation text
            ax.text(0.12, y_pos, rec['text'],
                   fontsize=self.fonts['body']['size'],
                   color=self.colors['dark'],
                   wrap=True,
                   verticalalignment='top')
            
            y_pos -= 0.15
    
    def create_footer(self, ax: plt.Axes, contact_info: Optional[str] = None) -> None:
        """
        Create infographic footer.
        
        Args:
            ax: Matplotlib axes
            contact_info: Optional contact information
        """
        ax.axis('off')
        
        # Footer line
        ax.plot([0, 1], [0.7, 0.7], 
               color=self.colors['primary'], linewidth=2)
        
        # UIDAI branding
        ax.text(0.05, 0.4, "UIDAI",
               fontsize=self.fonts['heading']['size'],
               fontweight='bold',
               color=self.colors['primary'])
        
        # Tagline
        ax.text(0.05, 0.2, "Data-Driven Insights for Better Service Delivery",
               fontsize=self.fonts['body']['size'],
               color=self.colors['neutral'],
               style='italic')
        
        # Contact info
        if contact_info:
            ax.text(0.95, 0.3, contact_info,
                   ha='right', va='center',
                   fontsize=self.fonts['body']['size'],
                   color=self.colors['neutral'])
    
    def generate_executive_summary_infographic(self, 
                                              data_summary: Dict[str, Any],
                                              filename: str = "executive_summary.png") -> str:
        """
        Generate complete one-page executive summary infographic.
        
        Args:
            data_summary: Dictionary containing all summary data
            filename: Output filename
            
        Returns:
            Path to saved infographic
        """
        # Create figure with custom layout
        fig = plt.figure(figsize=(11, 17))  # Letter size portrait
        gs = GridSpec(8, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Header (full width)
        ax_header = fig.add_subplot(gs[0, :])
        self.create_header(
            ax_header,
            title=data_summary.get('title', 'UIDAI Data Analysis'),
            subtitle=data_summary.get('subtitle', 'Executive Summary')
        )
        
        # Key metrics (3 cards)
        metrics = data_summary.get('key_metrics', [])
        for i, metric in enumerate(metrics[:3]):
            ax_metric = fig.add_subplot(gs[1, i])
            self.create_metric_card(
                ax_metric,
                value=metric.get('value', 'N/A'),
                label=metric.get('label', ''),
                icon=metric.get('icon', ''),
                color=metric.get('color', 'primary')
            )
        
        # Charts (2 columns)
        if 'bar_chart_data' in data_summary:
            ax_bar = fig.add_subplot(gs[2:4, 0:2])
            self.create_mini_bar_chart(
                ax_bar,
                data=data_summary['bar_chart_data'],
                title=data_summary.get('bar_chart_title', 'Distribution')
            )
        
        if 'pie_chart_data' in data_summary:
            ax_pie = fig.add_subplot(gs[2:4, 2])
            self.create_mini_pie_chart(
                ax_pie,
                data=data_summary['pie_chart_data'],
                title=data_summary.get('pie_chart_title', 'Composition')
            )
        
        # Key insights panel
        ax_insights = fig.add_subplot(gs[4:6, :])
        self.create_key_insights_panel(
            ax_insights,
            insights=data_summary.get('key_insights', [])
        )
        
        # Recommendations panel
        ax_recommendations = fig.add_subplot(gs[6, :])
        self.create_recommendations_panel(
            ax_recommendations,
            recommendations=data_summary.get('recommendations', [])
        )
        
        # Timeline
        if 'timeline_milestones' in data_summary:
            ax_timeline = fig.add_subplot(gs[7, :])
            self.create_timeline_visual(
                ax_timeline,
                milestones=data_summary['timeline_milestones']
            )
        
        # Footer
        # ax_footer = fig.add_subplot(gs[8, :])
        # self.create_footer(ax_footer, contact_info=data_summary.get('contact_info'))
        
        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Infographic saved: {output_path}")
        return str(output_path)


def create_sample_infographic():
    """Create sample executive infographic with demo data."""
    
    infographic = ExecutiveInfographic()
    
    # Sample data
    data_summary = {
        'title': 'UIDAI Data Hackathon 2026',
        'subtitle': 'Comprehensive Analysis of 987K+ Aadhaar Records',
        'key_metrics': [
            {'value': '987K+', 'label': 'Records Analyzed', 'icon': '📊', 'color': 'primary'},
            {'value': '300%', 'label': 'Projected ROI', 'icon': '💰', 'color': 'success'},
            {'value': '35%', 'label': 'Efficiency Gain', 'icon': '📈', 'color': 'secondary'}
        ],
        'bar_chart_data': {
            'Enrolment': 201,
            'Demographic': 414,
            'Biometric': 372
        },
        'bar_chart_title': 'Records by Type (Thousands)',
        'pie_chart_data': {
            'Children': 28.5,
            'Adults': 71.5
        },
        'pie_chart_title': 'Age Distribution',
        'key_insights': [
            "Identified 5 high-density geographic clusters accounting for 60% of updates",
            "Seasonal patterns show 40% higher activity in Q2, enabling proactive planning",
            "Children vs adults exhibit distinct behavioral patterns requiring tailored approaches",
            "Biometric-demographic correlation enables integrated service delivery optimization",
            "Predictive models achieve 92% accuracy in forecasting future update volumes"
        ],
        'recommendations': [
            {'text': 'Deploy predictive analytics for demand forecasting', 'priority': 'high'},
            {'text': 'Implement age-based segmentation for targeted outreach', 'priority': 'high'},
            {'text': 'Optimize resource allocation based on geographic clusters', 'priority': 'medium'},
            {'text': 'Establish real-time monitoring dashboard for operations', 'priority': 'medium'},
            {'text': 'Launch pilot program in top 3 high-density states', 'priority': 'high'}
        ],
        'timeline_milestones': [
            {'label': 'Q1: Planning'},
            {'label': 'Q2: Pilot'},
            {'label': 'Q3: Scale'},
            {'label': 'Q4: Optimize'}
        ],
        'contact_info': 'analytics@uidai.gov.in'
    }
    
    # Generate infographic
    output_path = infographic.generate_executive_summary_infographic(data_summary)
    print(f"✅ Executive infographic created: {output_path}")
    
    return output_path


def main():
    """Demonstration of infographic generation."""
    create_sample_infographic()


if __name__ == "__main__":
    main()
