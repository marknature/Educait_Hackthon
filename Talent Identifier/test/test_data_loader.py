import pytest
import pandas as pd
import sqlite3
import os
from pathlib import Path
from src.data_loader import load_sql_table, save_processed_data

@pytest.fixture
def test_db(tmp_path):
    """Create a temporary SQLite database for testing"""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    
    # Create test tables
    conn.execute("""
    CREATE TABLE players (
        id INTEGER PRIMARY KEY,
        player_name TEXT,
        position TEXT,
        market_value REAL
    )
    """)
    
    conn.execute("""
    CREATE TABLE player_attributes (
        player_id INTEGER,
        overall_rating REAL,
        FOREIGN KEY(player_id) REFERENCES players(id)
    )
    """)
    
    # Insert test data
    conn.executemany(
        "INSERT INTO players VALUES (?, ?, ?, ?)",
        [
            (1, "Player One", "DEFENDER", 10_000_000),
            (2, "Player Two", "MIDFIELDER", 20_000_000)
        ]
    )
    
    conn.executemany(
        "INSERT INTO player_attributes VALUES (?, ?)",
        [(1, 75.5), (2, 82.3)]
    )
    
    conn.commit()
    conn.close()
    return db_path

def test_load_sql_table(test_db):
    """Test loading data from SQL tables"""
    # Test loading players table
    players = load_sql_table("players", str(test_db))
    assert len(players) == 2
    assert "player_name" in players.columns
    assert players.iloc[0]["position"] == "DEFENDER"
    
    # Test loading attributes table
    attributes = load_sql_table("player_attributes", str(test_db))
    assert len(attributes) == 2
    assert "overall_rating" in attributes.columns

def test_save_processed_data(tmp_path):
    """Test saving processed data"""
    test_data = pd.DataFrame({
        "id": [1, 2],
        "value": [10, 20]
    })
    
    # Test saving as parquet
    parquet_path = save_processed_data(test_data, "test_data")
    assert os.path.exists(parquet_path)
    
    # Verify data integrity
    loaded = pd.read_parquet(parquet_path)
    pd.testing.assert_frame_equal(test_data, loaded)
    
    # Test saving as CSV
    csv_path = save_processed_data(test_data, "test_data", format="csv")
    assert os.path.exists(csv_path)
    loaded_csv = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(test_data, loaded_csv, check_dtype=False)

def test_load_nonexistent_table(test_db):
    """Test handling of missing tables"""
    with pytest.raises(Exception):
        load_sql_table("nonexistent", str(test_db))
