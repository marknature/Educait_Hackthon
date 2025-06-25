# 🧠 Talent Identification System – Educait Soccer Analytics Hackathon

## ⚽ Overview

This project is a machine learning-based **Talent Identification System** developed for **Challenge 1** of the **Educait Soccer Analytics Hackathon**. The goal is to help football scouts and managers identify **emerging talent** in the English Premier League by analyzing player data, development trajectories, and performance indicators.

## 📝 Problem Statement

*CHALLENGE 1**
Early identification of future football stars can provide teams with significant competitive and financial advantages. This system uses historical data and predictive modeling to:
- Spot high-potential players early in their careers
- Account for position, age, and team context
- Differentiate short-term performance from long-term growth
- Offer interpretable, scout-friendly insights

## 🧰 Technologies Used

- Python 3.8+
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, SHAP
- Jupyter Notebook
- (Optional: Streamlit for dashboard)

---

## 🔍 Approach

### 1. Data Preparation
- Loaded and cleaned player biographical and match statistics
- Handled missing data and standardized numerical features
- Encoded categorical variables like player position

### 2. Feature Engineering
- Created metrics like:
  - Age-adjusted performance scores
  - Improvement rate across seasons
  - Per-90-minute stats
  - Contextual stats based on team strength and minutes played

### 3. Modeling
- Trained a classification model (e.g., Random Forest, XGBoost) to predict future “stars”
- Used labeled data based on historical progression of known successful players

### 4. Evaluation
- Evaluated using Accuracy, F1-Score, and ROC-AUC
- Applied cross-validation to ensure robustness

### 5. Explainability & Visuals
- Feature importance visualized using SHAP
- Created radar charts, trend lines, and heatmaps for scout-friendly presentation

---

## 📊 Key Outputs

- A trained model that ranks emerging players by potential
- Visual dashboard for scouts and analysts
- Explainable predictions with key features driving each decision

---

## 🧪 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/educait-talent-id.git
   cd educait-talent-id
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch notebook or run scripts:

   ```bash
   jupyter notebook notebooks/talent_model.ipynb
   ```

(Optional) Run dashboard:

```bash
streamlit run app/dashboard.py
```

---

## 🎯 Evaluation Criteria Alignment

| Criteria                    | Description                            |
| --------------------------- | -------------------------------------- |
| 🧠 Feature Engineering      | Age-adjusted stats, trajectory metrics |
| 📊 Model Performance        | High accuracy, robust validation       |
| 💡 Explainability           | SHAP visuals + intuitive charts        |
| 📈 Real-World Applicability | Dashboard & scout-ready outputs        |

---

## 🎤 Presentation

A 5-minute slide presentation covers:

* Project background and motivation
* Feature and model selection
* Key insights
* Live demo (optional) of dashboard

---

## 🙌 Acknowledgments

* [Educait AI](https://educait.ai) for organizing the hackathon
* \[Zimbabwe Football Association (ZIFA)] for data support
* EPL open data sources and community contributors

---

## 📁 Project Structure

```
📦 soccer-analytics
├── data/
│   ├── raw/                    # Original SQL data files
│   │   └── database.sqlite
│   └── processed/              # Processed data (CSV/Parquet)
│       ├── match_stats.parquet
│       ├── player_metrics.parquet
│       └── player_events.parquet
├── notebooks/
│   └── talent_model.ipynb      # Jupyter notebook for exploration
├── src/
│   ├── data_loader.py          # SQL data loading utilities
│   ├── feature_engineering.py  # Feature creation
│   ├── modeling.py             # Model training/prediction
│   └── visualization.py       # Plotting functions
├── app/
│   ├── dashboard.py            # Streamlit/Plotly Dash app
│   └── templates/              # HTML templates (if web app)
├── report/
│   ├── technical_doc.pdf       # Methodology documentation
│   └── presentation.pdf        # Results presentation
├── tests/                      # Unit tests
│   └── test_features.py
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Future Work

* Expand to multi-league datasets
* Improve prediction of late bloomers
* Incorporate tracking data and scouting reports
