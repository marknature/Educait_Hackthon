import pytest
import pandas as pd
import os
from pathlib import Path
from src.visualization import SoccerVisualizer

@pytest.fixture
def sample_talent_data():
    """Create sample talent data for visualization tests"""
    return pd.DataFrame({
        "player_id": [1, 2, 3],
        "player_name": ["Player A", "Player B", "Player C"],
        "position": ["DEFENDER", "MIDFIELDER", "FORWARD"],
        "pes": [75, 82, 68],
        "log_market_value": [16.5, 17.2, 16.8],
        "market_value": [20_000_000, 30_000_000, 25_000_000],
        "undervalued": [0, 1, 0],
        "progressive_pass": [12, 18, 9],
        "tackles": [15, 8, 4],
        "dribbling": [5, 12, 15]
    })

@pytest.fixture
def sample_events():
    """Create sample event data for tests"""
    return pd.DataFrame({
        "player_id": [1, 1, 2, 2, 2, 3],
        "x_coord": [40, 35, 60, 65, 70, 80],
        "y_coord": [30, 40, 50, 45, 55, 40],
        "event_type": ["pass", "tackle", "pass", "dribble", "shot", "pass"]
    })

@pytest.fixture
def visualizer(tmp_path):
    """Initialize visualizer with temp output dir"""
    return SoccerVisualizer(output_dir=tmp_path / "figures")

def test_plot_talent_scatter(visualizer, sample_talent_data):
    """Test scatter plot generation"""
    fig = visualizer.plot_talent_scatter(sample_talent_data)
    assert fig is not None
    
    # Verify files were saved
    assert (visualizer.output_dir / "talent_scatter.png").exists()
    assert (visualizer.output_dir / "talent_scatter.pdf").exists()

def test_radar_chart(visualizer, sample_talent_data):
    """Test radar chart generation"""
    player_data = sample_talent_data.iloc[0]
    metrics = ["progressive_pass", "tackles", "dribbling"]
    
    fig = visualizer.plot_radar_chart(
        player_data,
        metrics=metrics,
        reference_data=sample_talent_data,
        title="Test Radar"
    )
    assert fig is not None
    assert (visualizer.output_dir / "test_radar.png").exists()

def test_heatmap(visualizer, sample_events):
    """Test positional heatmap generation"""
    fig = visualizer.plot_position_heatmap(
        sample_events,
        player_id=1,
        title="Test Heatmap"
    )
    assert fig is not None
    assert (visualizer.output_dir / "test_heatmap.png").exists()

def test_interactive_dashboard(visualizer, sample_talent_data, tmp_path):
    """Test interactive dashboard creation"""
    output_path = tmp_path / "dashboard.html"
    visualizer.create_interactive_dashboard(
        sample_talent_data,
        output_path=output_path
    )
    assert output_path.exists()
    assert os.path.getsize(output_path) > 0

def test_missing_data_handling(visualizer, sample_talent_data):
    """Test handling of missing/invalid data"""
    # Empty DataFrame
    with pytest.raises(ValueError):
        visualizer.plot_talent_scatter(pd.DataFrame())
    
    # Missing required columns
    with pytest.raises(KeyError):
        visualizer.plot_talent_scatter(
            sample_talent_data.drop(columns=["pes"])
        )
