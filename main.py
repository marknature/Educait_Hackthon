import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import cv2
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load player statistics data."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data for model training."""
    features = data[['goals', 'assists', 'cards', 'minutes_played', 'possession_metrics']]
    target = data['performance_rating']
    return features, target

def train_model(X_train, y_train):
    """Train the XGBoost model."""
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    return predictions

def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)

def analyze_momentum(video_path, events):
    """Analyze the video for emotional momentum shifts."""
    cap = cv2.VideoCapture(video_path)
    emotion_tags = []

    for index, event in events.iterrows():
        # Set the video position to the event time
        cap.set(cv2.CAP_PROP_POS_MSEC, event['time'] * 1000)
        ret, frame = cap.read()
        
        if ret:
            # Determine emotional state based on event type
            if event['type'] == 'goal':
                emotion_tags.append((event['time'], 'High Tension'))
            elif event['type'] == 'foul':
                emotion_tags.append((event['time'], 'Calm Control'))
            # Add more conditions as necessary

    cap.release()
    return emotion_tags

def plot_emotion_tags(emotion_tags):
    """Plot the emotional momentum shifts."""
    times, tags = zip(*emotion_tags)
    plt.figure(figsize=(10, 5))
    plt.scatter(times, [1]*len(times), c='red', label='High Tension', marker='o')
    plt.scatter(times, [0]*len(times), c='green', label='Calm Control', marker='x')
    plt.yticks([])
    plt.xlabel('Time (seconds)')
    plt.title('Emotional Momentum Shifts')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load player statistics data
    data = load_data('../data/player_stats.csv')  # Adjust the path as necessary

    # Preprocess data
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    predictions = evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model, '../outputs/player_performance_model.joblib')

    # Load match events data
    events = pd.read_csv('../data/match_events.csv')  # Adjust the path as necessary

    # Analyze momentum
    video_path = '../videos/match_video.mp4'  # Adjust the path as necessary
    emotion_tags = analyze_momentum(video_path, events)

    # Plot results
    plot_emotion_tags(emotion_tags)
