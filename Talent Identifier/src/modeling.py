"""
modeling.py - Talent identification modeling for soccer analytics
Includes PES calculation, anomaly detection, and validation methods
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import logging
import os
from src.modeling import TalentModeler
modeler = TalentModeler()
modeler.load_models("../notebooks/talent-identification-models.ipynb")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TalentModeler:
    def __init__(self, data_path='../data/processed/'):
        """Initialize model parameters and paths"""
        self.data_path = data_path
        self.position_weights = {
            'DEFENDER': {'progressive_pass': 0.3, 'progressive_dribble': 0.1, 
                        'x_coord': -0.2, 'tackles': 0.4},
            'MIDFIELDER': {'progressive_pass': 0.4, 'progressive_dribble': 0.3,
                          'x_coord': 0.1, 'pass_accuracy': 0.2},
            'FORWARD': {'progressive_pass': 0.2, 'progressive_dribble': 0.4,'x_coord': 0.3, 'shots_on_target': 0.4}
        }
        self.models = {}
        self.scaler = StandardScaler()
        
    def calculate_pes(self, features_df):
        """
        Calculate Position Efficiency Score (0-100 scale)
        
        Args:
            features_df: DataFrame with player features and position
            
        Returns:
            DataFrame with added 'pes' column
        """
        try:
            if 'position' not in features_df.columns:
                raise ValueError("Position column not found in input data")
                
            pes_scores = []
            for _, row in features_df.iterrows():
                weights = self.position_weights.get(row['position'], {})
                score = sum(row[col] * weight for col, weight in weights.items() if col in row)
                pes_scores.append(score)
                
            # Safe normalization
            max_score = max(pes_scores) if pes_scores else 1
            features_df['pes'] = (np.array(pes_scores) / max_score) * 100
            return features_df
            
        except Exception as e:
            logger.error(f"Error calculating PES: {str(e)}")
            raise

    def detect_talent_anomalies(self, features_df, market_values, contamination=0.1):
        """
        Identify undervalued players using isolation forest
        
        Args:
            features_df: DataFrame with player features including PES
            market_values: Series with player market values
            contamination: Expected fraction of anomalies
            
        Returns:
            DataFrame with anomaly scores and undervalued flags
        """
        try:
            if 'pes' not in features_df.columns:
                raise ValueError("PES column not found in features")
                
            X = pd.concat([
                features_df['pes'],
                np.log(market_values.replace(0, 1))  # Handle zero/negative values
            ], axis=1)
            
            clf = IsolationForest(n_estimators=200, 
                                contamination=contamination,
                                random_state=42)
            
            features_df['anomaly_score'] = clf.fit_predict(X)
            features_df['log_market_value'] = X.iloc[:, 1]
            
            # Label undervalued players (anomalies with high PES)
            median_pes = features_df['pes'].median()
            features_df['undervalued'] = np.where(
                (features_df['anomaly_score'] == -1) & 
                (features_df['pes'] > median_pes),
                1, 0
            )
            
            self.models['anomaly_detector'] = clf
            return features_df
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            raise
            
    def train_value_predictor(self, features_df, market_values, test_size=0.2):
        """
        Train a model to predict market value from features
        
        Args:
            features_df: DataFrame with player features
            market_values: Corresponding market values
            test_size: Fraction to hold out for testing
            
        Returns:
            Trained model and test metrics
        """
        try:
            X = self._prepare_features(features_df)
            y = np.log(market_values.replace(0, 1))
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            self.models['value_predictor'] = model
            logger.info(f"Trained value predictor with MSE: {mse:.3f}, R2: {r2:.3f}")
            
            return model, {'mse': mse, 'r2': r2}
            
        except Exception as e:
            logger.error(f"Error training value predictor: {str(e)}")
            raise
            
    def predict_value(self, player_features):
        """Predict market value for new player features"""
        try:
            if 'value_predictor' not in self.models:
                raise ValueError("Value predictor model not trained")
                
            X = self._prepare_features(player_features)
            return np.exp(self.models['value_predictor'].predict(X))
            
        except Exception as e:
            logger.error(f"Error predicting values: {str(e)}")
            raise
            
    def _prepare_features(self, features_df):
        """Prepare features for modeling (scaling, imputation, etc.)"""
        numeric_cols = features_df.select_dtypes(include=np.number).columns
        filled = features_df[numeric_cols].fillna(features_df[numeric_cols].median())
        return self.scaler.fit_transform(filled)
        
    def save_models(self, model_dir='models'):
        """Save trained models to disk"""
        os.makedirs(model_dir, exist_ok=True)
        for name, model in self.models.items():
            dump(model, f"{model_dir}/{name}.joblib")
        dump(self.scaler, f"{model_dir}/scaler.joblib")
        logger.info(f"Saved models to {model_dir}")
        
    def load_models(self, model_dir='models'):
        """Load trained models from disk"""
        model_files = {
            'anomaly_detector': f"{model_dir}/anomaly_detector.joblib",
            'value_predictor': f"{model_dir}/value_predictor.joblib",
            'scaler': f"{model_dir}/scaler.joblib"
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                self.models[name] = load(path)
            elif name != 'scaler':  # Scaler is optional
                logger.warning(f"Model file not found: {path}")
                
        logger.info("Loaded existing models")

    def calculate_metrics_with_opponent_adjustment(self, match_data, player_data):
        """
        Calculate opponent-adjusted performance metrics
        Args:
            match_data: DataFrame containing match details with opponent strength
            player_data: Player performance data to adjust
        Returns:
            DataFrame with additional opponent-adjusted metrics
        """
        try:
            # Merge match data with team ratings (assuming you have this)
            merged = player_data.merge(
                match_data[['match_id', 'opponent_strength']],
                on='match_id',
                how='left'
            )
            
            # Simple opponent strength adjustment
            merged['opponent_strength'] = merged['opponent_strength'].fillna(1.0)
            
            # Create adjusted versions of key metrics
            for metric in ['progressive_pass', 'tackles', 'shots_on_target']:
                if metric in merged.columns:
                    merged[f'adj_{metric}'] = merged[metric] * merged['opponent_strength']
            
            return merged
            
        except Exception as e:
            logger.error(f"Error calculating opponent adjustments: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    modeler = TalentModeler()
    
    # Load your processed data
    features = pd.read_parquet('../data/processed/player_features.parquet')
    market_values = pd.read_parquet('../data/processed/market_values.parquet')['value']
    
    # Train models
    scored_features = modeler.calculate_pes(features)
    anomaly_results = modeler.detect_talent_anomalies(scored_features, market_values)
    value_model, metrics = modeler.train_value_predictor(scored_features, market_values)
    
    # Save results
    anomaly_results.to_parquet('../data/processed/anomaly_results.parquet')
    modeler.save_models()
