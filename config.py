"""
Configuration constants for AltScore application.
DO NOT hardcode paths or values in main code - use this file.
"""

import os
from pathlib import Path

# Paths (relative to app.py)
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
VIZ_DIR = BASE_DIR / "visualizations"
REPORTS_DIR = BASE_DIR / "reports"

# Model files
MODEL_PATHS = {
    'lightgbm': MODELS_DIR / "lightgbm.pkl",
    'xgboost': MODELS_DIR / "xgboost.pkl",
    'random_forest': MODELS_DIR / "random_forest.pkl",
    'logistic_regression': MODELS_DIR / "logistic_regression.pkl"
}

# Performance metrics
MODEL_METRICS = {
    'lightgbm': {'auc': 0.7881, 'name': 'LightGBM'},
    'xgboost': {'auc': 0.7829, 'name': 'XGBoost'},
    'random_forest': {'auc': 0.7563, 'name': 'Random Forest'},
    'logistic_regression': {'auc': 0.7401, 'name': 'Logistic Regression'}
}

# Feature groups
TRAJECTORY_FEATURES = [
    'TRAJECTORY_EARLY_LATE_RATE',
    'TRAJECTORY_RECENT_LATE_RATE',
    'TRAJECTORY_SLOPE',
    'TRAJECTORY_IMPROVEMENT',
    'TRAJECTORY_SCORE'
]

BEHAVIORAL_FEATURES = [
    'BEHAVIORAL_CONSISTENCY',
    'CROSS_SOURCE_MEAN',
    'CROSS_SOURCE_RANGE'
]

NOVELTY_FEATURES = TRAJECTORY_FEATURES + BEHAVIORAL_FEATURES

# Fairness thresholds
FAIRNESS_THRESHOLD = 0.80  # DIR >= 0.80 is legal standard
FAIRNESS_ATTRIBUTES = ['Gender', 'Age', 'Income', 'Region', 'Credit History']

# UI Configuration
MAX_FILE_SIZE_MB = 200
DEFAULT_PAGE = "🏠 Home"
PAGES = [
    "🏠 Home",
    "💳 Credit Prediction",
    "📊 Model Comparison",
    "⚖️ Fairness Dashboard",
    "💡 Counterfactual Explanations"
]

# Counterfactual settings
ACTIONABLE_FEATURES = [
    'EXT_SOURCE_MEAN', 'DOCUMENT_COUNT', 'PAYMENT_DISCIPLINE_SCORE',
    'TRAJECTORY_SLOPE', 'BEHAVIORAL_CONSISTENCY'
]

# Team information
TEAM_INFO = {
    'name': 'TECH CRUSADERS',
    'members': [
        'Nithilan S - 3122245002060',
        'Muskan Kumari V - 3122245002059',
        'Charumadhi M - 3122245002012'
    ],
    'institution': 'Sri Sivasubramaniya Nadar College of Engineering',
    'department': 'Information Technology'
}

# Primary Colors
PRIMARY_BLUE = "#1E88E5"      # Main actions, headers
PRIMARY_DARK = "#1565C0"      # Hover states
PRIMARY_LIGHT = "#E3F2FD"     # Backgrounds

# Semantic Colors
SUCCESS_GREEN = "#4CAF50"     # Approved, pass
WARNING_ORANGE = "#FF9800"    # Borderline, caution
ERROR_RED = "#F44336"         # Rejected, fail
INFO_BLUE = "#2196F3"         # Information

# Neutral Colors
BACKGROUND = "#FFFFFF"
SECONDARY_BG = "#F5F7FA"
TEXT_PRIMARY = "#1A202C"
TEXT_SECONDARY = "#718096"
BORDER = "#E2E8F0"
