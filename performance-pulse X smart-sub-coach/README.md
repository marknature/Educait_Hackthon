# ⚽ Performance Pulse X Smart Sub Coach: AI-Powered Football Intelligence System

## 🚀 Overview

**Performance Pulse X Smart Sub Coach** is a comprehensive football analytics solution developed for the **Educait Soccer Analytics Hackathon 2.0**. This project combines two powerful AI components into one system:

- 🔮 **Performance Pulse**: Predicts individual player performance before a match using past performance data and context.
- 🧠 **Smart Sub Coach**: Provides real-time, intelligent substitution recommendations during a match based on performance dips, fatigue, and match dynamics.

Together, they empower coaches and analysts with **data-driven decision-making** tools for smarter lineups, strategic substitutions, and in-depth post-match analysis.

---

## 🎯 Problem Statement

In elite football, understanding performance isn't just about statistics—it's about **timing**, **momentum**, and **psychology**. Coaches must answer:

- Who is most likely to perform well in an upcoming match?
- When is the best moment to substitute a player?
- How do emotional momentum shifts impact the game?

This project addresses these questions through an AI-assisted pipeline that enhances tactical decisions with **predictive modeling**, **real-time analysis**, and **visual storytelling**.

---

## 💡 Solution Highlights

### 🧠 Module 1: Performance Pulse – Player Performance Prediction
- **Goal**: Forecast each player's expected performance rating.
- **Inputs**: Match stats, player history, opponent, location, cards, goals, assists.
- **Model**: Trained using Scikit-learn/XGBoost.
- **Output**: Player ratings + recommendation to **start**, **rest**, or **monitor**.

### 🔄 Module 2: Smart Sub Coach – Substitution Recommender
- **Goal**: Suggest real-time substitutions based on fatigue, fouls, and momentum.
- **Inputs**: In-game player stats (pass accuracy, tackles, distance, fouls, sprint rate).
- **Model**: Rule-based logic + ML-based substitution impact predictions.
- **Output**: Minute-by-minute substitution timeline + justification.

### 🔥 Bonus Feature (Optional): Emotional Momentum Timeline
- Analyzes match phases using stats and video to detect:
  - 🔴 High Tension
  - 🟢 Control
  - 🔵 Momentum Shift
- Helps coaches and analysts understand psychological game flow.

---

## 🛠 Tech Stack

| Category         | Tools / Libraries           |
|------------------|-----------------------------|
| Programming      | Python                      |
| Data Handling    | Pandas, NumPy               |
| ML Modeling      | Scikit-learn, XGBoost       |
| Visualization    | Matplotlib, Seaborn, Plotly |
| Video Processing | OpenCV, moviepy             |
| Dashboard (Optional) | Streamlit, Flask         |
| Audio (Optional) | librosa, pydub              |

---

## 📂 Project Structure

```bash
📁 performance-pulse-x-sub-coach/
│
├── 📁 data/                   # Player stats, match events
├── 📁 notebooks/              # Model building, EDA, exploration
├── 📁 src/
│   ├── performance_predictor.py
│   ├── substitution_recommender.py
│   └── momentum_detector.py   # (optional)
├── 📁 visualizations/         # Graphs, charts, timelines
├── 📁 videos/ (optional)      # Match footage clips
├── 📁 outputs/                # Predictions, recommendations
├── requirements.txt
├── README.md
└── app.py                    # Streamlit or Flask demo (optional)
````

---

## 📊 Sample Outputs

### ✅ Player Performance Prediction

```
Player: T. Alexander-Arnold
Predicted Rating: 7.4/10
Recommendation: START
Justification: Consistent high pass accuracy, strong history vs current opponent.
```

### 🔁 Substitution Suggestion

```
68' → SUB: Replace Midfielder #8 with #14
Reason: Accuracy dropped 88% → 62% | Yellow card | High opponent pressure
```

### 🧠 Momentum Timeline

```
05'–15'  🟢 Control (75% possession, low foul count)
22'–27'  🔴 High Tension (2 fouls + red card)
55'–65'  🔵 Momentum Shift (opponent pressing, 3 shots on target)
```

---

## 🧪 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/performance-pulse-x-sub-coach.git
   cd performance-pulse-x-sub-coach
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run performance predictor:

   ```bash
   python src/performance_predictor.py
   ```

4. Run substitution recommender:

   ```bash
   python src/substitution_recommender.py
   ```

5. (Optional) Launch dashboard:

   ```bash
   streamlit run app.py
   ```

---

## 🎥 Demo Video

📺 Watch our short demo on player performance prediction and substitution logic:
[🔗 Google Drive Folder](https://drive.google.com/drive/folders/1ggxvNFesKVJ6bTad4LE1sjg3IWn3IvYL?usp=drive_link)

---

## 👨‍💻 Team

* **Mark Chindudzi** – Machine Learning Engineer, Project Lead
  [GitHub](https://github.com/MarkNature) | [LinkedIn](https://www.linkedin.com/in/marknature-c)

---

## 📅 Submission Details

* **Hackathon**: Educait Soccer Analytics Hackathon – Campus Creator Challenge 2.0
* **Deadline**: June 24, 2025
* **Repository**: [GitHub Repo](https://github.com/educait-ai/soccer-analytics-hackthon)

---

## 🔮 Future Work

* Real-time data integration for live coaching
* Video evidence support for substitution suggestions
* Pose tracking for fatigue and injury risk
* Emotion and crowd audio signal modeling

---

🎯 *“Football is strategy. Data makes it smart.”*

```

---

Would you like this saved as a `.md` file or added directly to your GitHub repo? I can also help you create visuals or a Streamlit dashboard for the demo.
```
