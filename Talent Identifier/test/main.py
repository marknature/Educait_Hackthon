from src.data_loader import load_processed_data
from src.modeling import TalentModeler
from src.visualization import SoccerVisualizer

def main():
  # Load data
  player_data = load_processed_data('player_data.parquet')
  # Initialize modeler
  modeler = TalentModeler()
  # Train models
  # Visualize results
  # Save outputs

if __name__ == "__main__":
  main()
