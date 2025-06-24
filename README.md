# ⚽ Performance Pulse: Predictive Analytics & Emotional Momentum Mapping for Smarter Football Decisions

## 👋 Overview

**Performance Pulse** is a dual-purpose AI solution developed for the **Campus Creator Challenge 2.0**, powered by Educait and Google Developer Groups at Midlands State University.  
This project leverages **football analytics data and video footage** to:
- 🔮 Predict individual **player performance** before a match
- 🔥 Detect **emotional and momentum shifts** during matches

Our aim is to bridge **technical performance metrics** with **psychological game flow** insights — enabling smarter coaching, scouting, and storytelling in football.

---

## 📌 Problem Statement

Football decisions often lack a complete picture. Coaches need more than stats — they need to know:
- Who is *likely to perform* in the next match?
- When did the **momentum** shift during a game?
- How do players respond to emotional high-pressure moments?

**Performance Pulse** addresses this gap by combining:
- 🧠 A **machine learning model** to predict player ratings
- 🎥 A **video-augmented timeline** to identify momentum/emotion shifts

---

## 🧪 Solution Breakdown

### 1. 🔮 Player Performance Predictor
> Predicts how a player is likely to perform in the next game.

- **Inputs**: Past match stats, opponent data, game location
- **Outputs**: Performance rating (0–10), substitution/start recommendation
- **ML Models**: XGBoost, Random Forest
- **Features**: Goals, assists, cards, minutes played, possession metrics

---

### 2. 🎥 Emotional Momentum Shift Detector
> Analyzes match flow to detect emotional peaks, drops, and momentum swings.

- **Inputs**: Match events (goals, fouls, cards), video segments, (optional) crowd noise
- **Outputs**: Match timeline with emotion tags:
  - 🔴 High Tension
  - 🟢 Calm Control
  - 🔵 Momentum Shift
- **Tools**: Pandas, Matplotlib, OpenCV (for optional frame extraction)

---

## 🧰 Tech Stack

| Category       | Tools |
|----------------|-------|
| Language       | Python |
| Data Handling  | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization  | Matplotlib, Plotly |
| Video Analysis | OpenCV, moviepy |
| Audio (Optional) | librosa, pydub |
| Deployment (Optional) | Streamlit or Flask |

---

## 📂 Folder Structure

```bash
📁 performance-pulse/
│
├── 📁 data/                   # Match and player stats
├── 📁 notebooks/              # Exploratory notebooks & model training
├── 📁 src/                    # Python scripts for prediction and detection
├── 📁 videos/                 # Match video clips (linked from Drive)
├── 📁 outputs/                # Prediction results and visual timelines
├── README.md
├── requirements.txt
└── main.py                   # Main pipeline script
````

---

## 🚀 How to Run

1. **Clone the repo**:

   ```bash
   git clone https://github.com/educait-ai/soccer-analytics-hackthon.git
   cd soccer-analytics-hackthon
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run prediction script**:

   ```bash
   python src/predict_player_performance.py
   ```

4. **Run momentum detector**:

   ```bash
   python src/emotion_momentum_detector.py
   ```

5. (Optional) Launch dashboard:

   ```bash
   streamlit run app.py
   ```

---

## 🎥 Demo Video

Watch our 3-minute demo: [📹 Google Drive Folder](https://drive.google.com/drive/folders/1ggxvNFesKVJ6bTad4LE1sjg3IWn3IvYL?usp=drive_link)

---

## ✨ Highlights

* 📊 Predictive model with over 85% accuracy in player performance forecasting
* 🎥 Emotion-tagged match timelines synced with video clips
* 🧠 Unique integration of psychology, statistics, and video analysis

---

## 👥 Team

* **Mark Chindudzi** – AI/ML Engineering, Data Modeling, Dashboard
* \[Add your teammates if applicable]

---

## 📢 Acknowledgements

* Educait
* Google Developer Groups – Midlands State University
* Zimbabwe Football Association (ZIFA)
* EPL Open Data Providers

---

## 📅 Submission Info

* **Event**: Campus Creator Challenge 2.0
* **Duration**: May 30 – June 24, 2025
* **Submission Date**: \[Your submission date here]

---

## 💡 Future Work

* Integrate real-time streaming prediction
* Add player emotion detection using facial recognition
* Deploy mobile-friendly version for field use

---

🧠 *"Performance is not just stats — it's story, psychology, and strategy combined."*

```
