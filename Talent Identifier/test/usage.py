from src.visualization import SoccerVisualizer
from src.data_loader import load_processed_data

# Initialize with output directory
viz = SoccerVisualizer(output_dir="reports/figures")

# Load your data
talent_data = load_processed_data('talent_metrics.parquet')

# Create standard scatter plot
viz.plot_talent_scatter(
    talent_data,
    title="Top League Talent Analysis",
    save_name="talent_analysis"
)

# Generate interactive dashboard
viz.create_interactive_dashboard(
    talent_data,
    output_path="reports/interactive_dashboard.html"
)
# Note: Ensure that the 'talent_metrics.parquet' file exists in the expected directory.
# The above code assumes that the 'src.visualization' and 'src.data_loader' modules
