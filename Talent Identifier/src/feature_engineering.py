import numpy as np

def calculate_progressive_passes(
    events_df,
    min_progress=15, 
    x_col='start_x',
    end_x_col='end_x'
):
    """Calculate progressive passes (net forward movement > min_progress)"""
    progress = events_df[end_x_col] - events_df[x_col]
    return events_df.assign(
        progressive_pass=(progress > min_progress).astype(int)
    )

def calculate_xG(shot_df):
    """Calculate expected goals based on shot characteristics"""
    conditions = [
        (shot_df['shot_type'] == 'header'),
        (shot_df['shot_distance'] < 18),
        (shot_df['shot_body_part'] == 'foot')
    ]
    weights = [0.1, 0.15, 0.25]  # Example weights
    return np.select(conditions, weights, default=0.05)
