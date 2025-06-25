import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_loader import load_sql_table

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    player_data = load_sql_table('players')
    match_data = load_sql_table('match_stats')
    return player_data.merge(match_data, on='player_id')

def main():
    st.title("Soccer Performance Analytics Dashboard")
    
    df = load_data()
    
    st.sidebar.header("Filters")
    position_filter = st.sidebar.multiselect(
        "Player Position",
        options=df['position'].unique()
    )
    
    filtered_df = df[df['position'].isin(position_filter)] if position_filter else df
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Player Efficiency Scores")
        fig = px.scatter(
            filtered_df,
            x='efficiency_score',
            y='market_value',
            color='position',
            hover_data=['player_name']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("Progressive Impact by Position")
        fig = px.box(
            filtered_df,
            x='position',
            y='progressive_passes_p90',
            color='position'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
# This code is a Streamlit dashboard for visualizing soccer player performance data.
