"""
Visualization tools for soccer talent analytics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
import logging
import os
from pathlib import Path

# Configure plotting style
plt.style.use('seaborn')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SoccerVisualizer:
    def __init__(self, output_dir: str = "reports/figures"):
        """
        Initialize visualizer with output directory
        
        Args:
            output_dir: Path to save generated visualizations
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.custom_colors = {
            'undervalued': '#e74c3c',
            'normal': '#3498db',
            'highlight': '#f1c40f'
        }
    
    def plot_talent_scatter(
        self,
        data: pd.DataFrame,
        x_col: str = 'log_market_value',
        y_col: str = 'pes',
        hue_col: str = 'undervalued',
        style_col: Optional[str] = 'position',
        title: str = "Talent Identification",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create scatter plot of player efficiency vs market value
        
        Args:
            data: DataFrame containing player metrics
            x_col: Column for x-axis (typically market value)
            y_col: Column for y-axis (typically PES)
            hue_col: Column to color points by
            style_col: Column to vary point markers by
            title: Plot title
            save_name: Filename to save plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Create the scatter plot
            scatter = sns.scatterplot(
                data=data,
                x=x_col,
                y=y_col,
                hue=hue_col,
                style=style_col if style_col in data.columns else None,
                palette=self.custom_colors if hue_col == 'undervalued' else None,
                s=100,
                alpha=0.8,
                ax=ax
            )
            
            # Add median line
            if y_col in data.columns:
                median_val = data[y_col].median()
                ax.axhline(
                    y=median_val, 
                    color='black', 
                    linestyle='--',
                    label=f'Median {y_col}'
                )
            
            # Formatting
            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
            
            # Improve legend
            handles, labels = ax.get_legend_handles_labels()
            if style_col in data.columns:
                # Separate hue and style legends
                n_hue = len(data[hue_col].unique()) if hue_col in data.columns else 0
                ax.legend(
                    handles[:n_hue] + handles[-1:],  # Show hue legend and median line
                    labels[:n_hue] + labels[-1:],
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left'
                )
            else:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
            plt.tight_layout()
            
            if save_name:
                self._save_figure(fig, save_name)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            raise
    
    def plot_radar_chart(
        self,
        player_data: Union[pd.DataFrame, Dict],
        metrics: List[str],
        reference_data: Optional[pd.DataFrame] = None,
        title: str = "Player Profile",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create radar chart comparing player attributes
        
        Args:
            player_data: DataFrame row or dict of player metrics
            metrics: List of metric names to include in radar
            reference_data: Optional DataFrame for comparison (e.g., position average)
            title: Chart title
            save_name: Filename to save plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Prepare data for radar plot
            if isinstance(player_data, pd.DataFrame):
                player_data = player_data.iloc[0].to_dict()
                
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Close the circle
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
            
            # Plot player values
            values = [player_data[m] for m in metrics]
            values += values[:1]  # Close the circle
            ax.plot(angles, values, linewidth=2, label=player_data.get('player_name', 'Player'))
            ax.fill(angles, values, alpha=0.25)
            
            # Plot reference if provided
            if reference_data is not None:
                ref_values = [reference_data[m].mean() for m in metrics]
                ref_values += ref_values[:1]
                ax.plot(angles, ref_values, linewidth=2, linestyle='--', label=f"{reference_data['position'].iloc[0]} Avg")
            
            # Formatting
            ax.set_theta_offset(np.pi/2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
            
            ax.set_title(title, pad=20, fontsize=16)
            ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
            
            if save_name:
                self._save_figure(fig, save_name)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {str(e)}")
            raise
    
    def plot_metrics_over_time(
        self,
        time_series_data: pd.DataFrame,
        player_id: int,
        metrics: List[str],
        rolling_window: int = 5,
        title: str = "Performance Trend",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot metric trends over time with rolling average
        
        Args:
            time_series_data: DataFrame with datetime index and player metrics
            player_id: ID of player to plot
            metrics: List of metrics to include
            rolling_window: Window for rolling average
            title: Plot title
            save_name: Filename to save plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        try:
            player_data = time_series_data[time_series_data['player_id'] == player_id]
            if player_data.empty:
                raise ValueError(f"No data found for player_id {player_id}")
                
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for metric in metrics:
                if metric in player_data.columns:
                    # Plot raw values
                    ax.plot(
                        player_data.index,
                        player_data[metric],
                        alpha=0.3,
                        label=f'{metric} (raw)'
                    )
                    
                    # Plot rolling average
                    ax.plot(
                        player_data.index,
                        player_data[metric].rolling(rolling_window).mean(),
                        linewidth=2,
                        label=f'{metric} ({rolling_window}-game avg)'
                    )
            
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Date")
            ax.set_ylabel("Metric Value")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_name:
                self._save_figure(fig, save_name)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {str(e)}")
            raise
    
    def plot_position_heatmap(
        self,
        event_data: pd.DataFrame,
        player_id: int,
        pitch_dimensions: tuple = (100, 100),
        title: str = "Positioning Heatmap",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create heatmap of player's positional data
        
        Args:
            event_data: DataFrame containing event coordinates
            player_id: ID of player to analyze
            pitch_dimensions: Tuple of (length, width) for pitch
            title: Plot title
            save_name: Filename to save plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        try:
            player_events = event_data[event_data['player_id'] == player_id]
            if player_events.empty:
                raise ValueError(f"No event data found for player_id {player_id}")
                
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Create heatmap
            sns.kdeplot(
                data=player_events,
                x='x_coord',
                y='y_coord',
                fill=True,
                cmap='viridis',
                thresh=0.1,
                alpha=0.8,
                ax=ax
            )
            
            # Draw pitch boundaries
            ax.set_xlim(0, pitch_dimensions[0])
            ax.set_ylim(0, pitch_dimensions[1])
            ax.axhline(y=pitch_dimensions[1]/2, color='black')  # Halfway line
            
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Pitch Length (yards)")
            ax.set_ylabel("Pitch Width (yards)")
            
            if save_name:
                self._save_figure(fig, save_name)
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            raise
    
    def create_interactive_dashboard(
        self, 
        talent_data: pd.DataFrame, 
        output_path: str = "reports/dashboard.html"
    ) -> None:
        """
        Create an interactive HTML dashboard using Plotly
        
        Args:
            talent_data: Complete talent analysis DataFrame
            output_path: Path to save HTML output
        """
        try:
            if talent_data.empty:
                raise ValueError("No talent data provided")
                
            # Create interactive scatter plot
            scatter_fig = px.scatter(
                talent_data,
                x='log_market_value',
                y='pes',
                color='undervalued',
                hover_data=['player_name', 'position', 'market_value'],
                title="Player Talent Identification",
                color_discrete_map={0: self.custom_colors['normal'], 1: self.custom_colors['undervalued']}
            )
            
            # Create parallel coordinates plot for attributes
            numeric_cols = talent_data.select_dtypes(include=np.number).columns.tolist()
            parallel_fig = px.parallel_coordinates(
                talent_data[numeric_cols + ['position']],
                color='pes',
                title="Attribute Relationships"
            )
            
            # Combine into dashboard
            from plotly.subplots import make_subplots
            dash_fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Player Valuation", "Attribute Profile")
            )
            
            dash_fig.add_trace(scatter_fig.data[0], row=1, col=1)
            dash_fig.add_trace(scatter_fig.data[1], row=1, col=1)
            dash_fig.add_trace(parallel_fig.data[0], row=2, col=1)
            
            dash_fig.update_layout(height=1000, showlegend=True)
            
            # Save to HTML
            dash_fig.write_html(output_path)
            logger.info(f"Saved interactive dashboard to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Internal method to save figures with consistent formatting
        
        Args:
            fig: matplotlib Figure object
            filename: Output filename (without extension)
        """
        formats = ['png', 'pdf']
        for fmt in formats:
            path = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved figures to {self.output_dir}/{filename}.[png|pdf]")

if __name__ == "__main__":
    # Example usage
    from src.data_loader import load_processed_data
    
    visualizer = SoccerVisualizer()
    
    # Load sample data (replace with your actual data loading)
    talent_data = load_processed_data('talent_metrics.parquet')
    events_data = load_processed_data('player_events.parquet')
    
    # Generate visualizations
    visualizer.plot_talent_scatter(
        talent_data,
        title="Premier League Talent Identification",
        save_name="talent_scatter"
    )
    
    # Show example player radar
    sample_player = talent_data.iloc[0]
    visualizer.plot_radar_chart(
        sample_player,
        metrics=['progressive_pass', 'tackles', 'shots_on_target', 'dribbling'],
        reference_data=talent_data[talent_data['position'] == sample_player['position']],
        title=f"Player Profile: {sample_player['player_name']}",
        save_name="sample_radar"
    )
    
    # Create interactive dashboard
    visualizer.create_interactive_dashboard(talent_data)
