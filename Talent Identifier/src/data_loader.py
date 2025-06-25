import sqlite3
import pandas as pd
import os

def load_sql_table(table_name, db_path='data/raw/database.sqlite'):
    """Load a table from SQLite database into a DataFrame"""
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def save_processed_data(df, filename, format='parquet'):
    """Save processed data to processed directory"""
    os.makedirs('data/processed', exist_ok=True)
    path = f'data/processed/{filename}'
    if format == 'parquet':
        df.to_parquet(path + '.parquet')
    else:
        df.to_csv(path + '.csv', index=False)
    return path
