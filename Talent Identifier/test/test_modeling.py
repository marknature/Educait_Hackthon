import pytest
import pandas as pd
import numpy as np
from src.modeling import TalentModeler

@pytest.fixture
def sample_data():
    """Create sample player data for testing"""
    return pd.DataFrame({
        "player_id": [1, 2, 3],
        "position": ["DEFENDER", "MIDFIELDER", "FORWARD"],
        "progressive_pass": [10, 15, 8],
        "progressive_dribble": [2, 8, 12],
        "x_coord": [0.4, 0.5, 0.6],
        "tackles": [20, 10, 5],
        "pass_accuracy": [0.7, 0.8, 0.75],
        "shots_on_target": [1, 3, 5]
    })

@pytest.fixture
def modeler():
    """Initialize a clean modeler instance"""
    return TalentModeler()

def test_calculate_pes(modeler, sample_data):
    """Test PES calculation"""
    scored = modeler.calculate_pes(sample_data)
    
    # Validate outputs
    assert "pes" in scored.columns
    assert all(0 <= pes <= 100 for pes in scored["pes"])
    assert not scored["pes"].isnull().any()
    
    # Verify position-specific calculations
    defender_pes = scored[scored["position"] == "DEFENDER"]["pes"].values[0]
    mid_pes = scored[scored["position"] == "MIDFIELDER"]["pes"].values[0]
    assert defender_pes != mid_pes  # Different positions should have different scores

def test_detect_talent_anomalies(modeler, sample_data):
    """Test anomaly detection"""
    # Add market values
    market_values = pd.Series([10_000_000, 50_000_000, 30_000_000])
    
    # First calculate PES
    scored = modeler.calculate_pes(sample_data)
    
    # Detect anomalies
    result = modeler.detect_talent_anomalies(scored, market_values)
    
    # Validate outputs
    assert "anomaly_score" in result.columns
    assert "undervalued" in result.columns
    assert result["undervalued"].isin([0, 1]).all()
    
    # Test with different contamination
    result_low = modeler.detect_talent_anomalies(scored, market_values, contamination=0.01)
    assert result_low["undervalued"].sum() <= result["undervalued"].sum()

def test_value_predictor(modeler, sample_data):
    """Test market value prediction"""
    market_values = pd.Series([10_000_000, 20_000_000, 15_000_000])
    
    # Train model
    model, metrics = modeler.train_value_predictor(sample_data, market_values)
    
    # Validate training
    assert "value_predictor" in modeler.models
    assert isinstance(metrics["mse"], float)
    assert 0 <= metrics["r2"] <= 1
    
    # Test predictions
    predictions = modeler.predict_value(sample_data)
    assert len(predictions) == len(sample_data)
    assert all(predictions > 0)

def test_save_load_models(modeler, sample_data, tmp_path):
    """Test model serialization"""
    # Train a model first
    market_values = pd.Series([10_000_000, 20_000_000, 15_000_000])
    modeler.train_value_predictor(sample_data, market_values)
    
    # Save models
    model_dir = tmp_path / "models"
    modeler.save_models(model_dir)
    
    # Verify files exist
    assert (model_dir / "value_predictor.joblib").exists()
    assert (model_dir / "scaler.joblib").exists()
    
    # Load into new modeler
    new_modeler = TalentModeler()
    new_modeler.load_models(model_dir)
    
    # Verify model works
    assert hasattr(new_modeler.models["value_predictor"], "predict")
    predictions = new_modeler.predict_value(sample_data)
    assert len(predictions) == len(sample_data)
