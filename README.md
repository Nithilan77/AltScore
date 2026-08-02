# 🏆 AltScore - Fair Credit Scoring System

<div align="center">

![AltScore](https://img.shields.io/badge/AltScore-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

**Making Credit Access Fair for Everyone**

[Documentation](#documentation) • [Team](#team)

</div>

---

## 🎯 Overview

AltScore is a production-grade, fair, and explainable credit scoring system that:

- 🎯 **Predicts** default risk with 0.7881 ROC-AUC (LightGBM champion)
- 💡 **Explains** decisions through counterfactual recommendations (88.9% validation rate)
- ⚖️ **Ensures** fairness across demographics (4/5 attributes pass DIR ≥ 0.80)
- 📈 **Rewards** improving behavior through financial trajectory analysis

Built for **Zenith Hackathon 5.0** by Team TECH CRUSADERS.

---

## ✨ Key Innovations

### 1. Behavioral Consistency Analysis
Measures payment consistency across 3 data sources (bureau, POS, installments) to catch applicants hiding bad behavior.

**Impact:** Ranked #17 out of 239 features (top 10%)

### 2. Financial Trajectory Scoring
Tracks whether payment behavior is improving or declining over time, rewarding progress.

**Impact:** Features rank #61 and #73 (top 30%)

### 3. Fairness-Aware Analysis
Ensures no demographic discrimination through rigorous bias audits using Disparate Impact Ratio.

**Impact:** Identified age bias before deployment (DIR 0.649)

### 4. Counterfactual Explanations
Shows rejected applicants EXACTLY what to change to get approved using optimization algorithms.

**Impact:** 88.9% validation success rate (vs 6% baseline)

---

## 📊 Model Performance

| Model | ROC-AUC | Status |
|-------|---------|--------|
| **LightGBM** | **0.7881** | 🏆 **Champion** |
| XGBoost | 0.7829 | ✅ Excellent |
| Random Forest | 0.7563 | ✅ Strong |
| Logistic Regression | 0.7401 | ✅ Baseline |

**Dataset:** 307,511 applicants × 244 features  
**Novel Features:** 6 (5 trajectory + 1 behavioral)  
**Target Exceeded:** +3.8% above 0.75 goal

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pip
```

### Installation
```bash
# Clone repository
git clone https://github.com/Nithilan77/AltScore.git
cd AltScore

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

App will open at `http://localhost:8501`

---

## 📁 Project Structure
```
altscore-hackathon/
├── app.py                      # Main Streamlit application
├── utils.py                    # Helper functions
├── config.py                   # Configuration constants
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .streamlit/
│   └── config.toml            # Streamlit theme configuration
├── models/
│   └── lightgbm.pkl           # Champion model (150MB)
├── visualizations/
│   ├── roc_curves_comparison.png
│   ├── feature_importance_top20.png
│   ├── fairness_dashboard.png
│   └── ...
├── reports/
│   ├── feature_importance.csv
│   └── fairness_metrics_summary.csv
├── data/
│   ├── sample_applicant.csv   # Sample data for testing
│   └── feature_metadata.json  # Feature documentation
└── tests/
    └── manual_test_checklist.md
```

---

## 🛠️ Technology Stack

**Machine Learning:**
- LightGBM 4.3.0 (Champion)
- XGBoost 2.0.3
- scikit-learn 1.4.0

**Data Processing:**
- pandas 2.1.4
- numpy 1.26.3

**Web Application:**
- Streamlit 1.31.0

**Visualization:**
- matplotlib 3.8.2
- seaborn 0.13.1

---

## 📖 Documentation

### Feature Engineering

**6 Novel Features Created:**

1. `TRAJECTORY_EARLY_LATE_RATE` - Late payment rate in first half of history
2. `TRAJECTORY_RECENT_LATE_RATE` - Late payment rate in recent half (Rank #61)
3. `TRAJECTORY_SLOPE` - Linear trend of payment behavior (Rank #73)
4. `TRAJECTORY_IMPROVEMENT` - Boolean flag for improvement
5. `TRAJECTORY_SCORE` - Composite score (Rank #115)
6. `BEHAVIORAL_CONSISTENCY` - Cross-source payment consistency (Top 10%)

### Model Training

**LightGBM Hyperparameters:**
```python
{
    'n_estimators': 1000,
    'learning_rate': 0.02,
    'max_depth': 7,
    'scale_pos_weight': 11,
    'random_state': 42
}
```

**Cross-Validation:** 5-fold Stratified, ROC-AUC: 0.7827 ± 0.012

### API Reference
```python
import joblib
import pandas as pd

# Load model
model = joblib.load("models/lightgbm.pkl")

# Predict
applicant_data = pd.read_csv("applicant.csv")  # 244 features required
prediction = model.predict(applicant_data)[0]  # 0=approve, 1=reject
probability = model.predict_proba(applicant_data)[0, 1]  # default probability

# Interpret
approval_prob = 1 - probability
decision = "APPROVED" if prediction == 0 else "REJECTED"
```

**Required Features:** See `reports/feature_importance.csv` for complete list

---

## ⚖️ Fairness Analysis

**Disparate Impact Ratios (DIR):**

| Attribute | DIR | Status | Issue |
|-----------|-----|--------|-------|
| Gender | 0.828 | ✅ PASS | None |
| **Age** | **0.649** | ❌ **FAIL** | 25% approval gap |
| Income | 0.869 | ✅ PASS | None |
| Region | 0.805 | ✅ PASS | Borderline |
| Credit History | 1.204 | ✅ PASS | Favors thin-file |

**Legal Standard:** DIR ≥ 0.80 (80% rule)

**Mitigation Plan:**
1. Reweight training samples by age group
2. Apply fairness constraints (AIF360)
3. Post-processing calibration
4. Remove age-correlated proxy features

---

## 💡 Counterfactual Explanations

**Validation Results:**
- **Success Rate:** 88.9% (24/27 applicants)
- **Avg Probability Gain:** +20.4 percentage points
- **Method:** Hybrid optimization (L-BFGS-B + Genetic)

**Example:**
```
Applicant #4: 22.3% → 51.7% approval probability
Changes Required:
1. EXT_SOURCE_MEAN: 0.45 → 0.60 (EASY, 3-6 months)
2. DOCUMENT_COUNT: 3 → 7 (VERY EASY, immediate)
3. PAYMENT_DISCIPLINE_SCORE: 0.70 → 1.20 (EASY, 6 months)
Result: REJECTED → APPROVED ✅
```

---

## 👥 Team

**Team TECH CRUSADERS**

- **Nithilan S** - 3122245002060  
  *Lead Developer | Counterfactual Explanations*

- **Muskan Kumari V** - 3122245002059  
  *ML Engineer | Financial Trajectory & Behavioral Consistency*

- **Charumadhi M** - 3122245002012  
  *Data Scientist | Fairness Analysis*

**Institution:** Sri Sivasubramaniya Nadar College of Engineering  
**Department:** Information Technology  
**Event:** Zenith Hackathon 5.0 - March 2026

---

## 📄 License

This project was developed for educational purposes as part of Zenith Hackathon 5.0.

---

## 🙏 Acknowledgments

- **Dataset:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)
- **Inspiration:** DICE-ML library for counterfactual explanations
- **Fairness Metrics:** Fairlearn library principles
- **Platform:** Streamlit for rapid deployment

---

## 📞 Contact

- **GitHub:** [github.com/Nithilan77/AltScore](https://github.com/Nithilan77/AltScore)
- **Issues:** [Report a bug](https://github.com/Nithilan77/AltScore/issues)

---

<div align="center">

**Built with ❤️ by Team TECH CRUSADERS**

*Making Credit Access Fair for Everyone*

</div>
