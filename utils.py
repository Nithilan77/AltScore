"""
Utility functions for AltScore application.
All reusable logic should be here, not in app.py.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import dice_ml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelWrapper:
    def __init__(self, m):
        self.model = m
    def predict(self, df):
        proba = self.model.predict_proba(df)[:, 1]
        return (proba < 0.5).astype(int)
    def predict_proba(self, df):
        probs = self.model.predict_proba(df)
        prob_approved = 1.0 - probs[:, 1]
        prob_rejected = probs[:, 1]
        return np.vstack((prob_rejected, prob_approved)).T

@st.cache_resource(show_spinner="Loading model...")
def load_model(model_name: str = 'lightgbm'):
    """
    Load ML model with caching for performance.
    """
    import config
    
    model_path = config.MODEL_PATHS.get(model_name)
    
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found at {model_path}")
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded {model_name} model")
        return model
    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}")
        raise

@st.cache_data(show_spinner="Loading feature metadata...")
def load_feature_metadata() -> Dict:
    import json
    import config
    path = config.DATA_DIR / "feature_metadata.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner="Loading feature importance...")
def load_feature_importance() -> pd.DataFrame:
    """Load and cache feature importance data."""
    import config
    
    path = config.REPORTS_DIR / "feature_importance.csv"
    if not path.exists():
        logger.warning("Feature importance file not found")
        return pd.DataFrame()
    
    return pd.read_csv(path)

@st.cache_data(show_spinner="Loading fairness metrics...")
def load_fairness_metrics() -> pd.DataFrame:
    """Load and cache fairness metrics."""
    import config
    
    path = config.REPORTS_DIR / "fairness_metrics_summary.csv"
    if not path.exists():
        logger.warning("Fairness metrics file not found")
        return pd.DataFrame()
    
    return pd.read_csv(path)

def validate_uploaded_file(uploaded_file) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded CSV file.
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    if not uploaded_file.name.endswith('.csv'):
        return False, "File must be CSV format"
    
    import config
    max_size = config.MAX_FILE_SIZE_MB * 1024 * 1024
    
    if uploaded_file.size > max_size:
        return False, f"File size exceeds {config.MAX_FILE_SIZE_MB}MB limit"
    
    return True, None

def validate_features(df: pd.DataFrame, required_features: List[str]) -> Tuple[bool, List[str]]:
    """Check if DataFrame has all required features."""
    missing = list(set(required_features) - set(df.columns))
    return len(missing) == 0, missing

def format_probability(prob: float, decimals: int = 1) -> str:
    """Format probability as percentage string."""
    return f"{prob * 100:.{decimals}f}%"

def get_decision_emoji(prediction: int) -> str:
    """Return emoji based on prediction."""
    return "✅" if prediction == 0 else "❌"

def get_decision_text(prediction: int) -> str:
    """Return decision text based on prediction."""
    return "APPROVED" if prediction == 0 else "REJECTED"

def get_decision_color(prediction: int) -> str:
    """Return color based on prediction."""
    import config
    return config.SUCCESS_GREEN if prediction == 0 else config.ERROR_RED

def create_metric_card(title: str, value: str, delta: Optional[str] = None, 
                      color: str = "#1E88E5") -> None:
    """Create a styled metric card."""
    st.markdown(f'''
    <div style="
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid {color};
        margin-bottom: 1rem;
    ">
        <p style="color: #6c757d; font-size: 0.875rem; margin: 0;">{title}</p>
        <h2 style="color: #212529; margin: 0.5rem 0;">{value}</h2>
        {f'<p style="color: {color}; font-size: 0.875rem; margin: 0;">{delta}</p>' if delta else ''}
    </div>
    ''', unsafe_allow_html=True)

def show_error(message: str) -> None:
    """Display styled error message."""
    st.markdown(f'''
    <div style="
        background-color: #fee;
        border: 1px solid #fcc;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #c33;
    ">
        ⚠️ <strong>Error:</strong> {message}
    </div>
    ''', unsafe_allow_html=True)

def show_success(message: str) -> None:
    """Display styled success message."""
    st.markdown(f'''
    <div style="
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #155724;
    ">
        ✅ <strong>Success:</strong> {message}
    </div>
    ''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def generate_sample_applicant() -> pd.DataFrame:
    """Generate or load a sample applicant."""
    import config
    sample_path = config.DATA_DIR / "sample_applicant.csv"
    
    if sample_path.exists():
        return pd.read_csv(sample_path)
    
    logger.warning("Sample data file not found. Wait for data population step.")
    return pd.DataFrame()

def get_feature_description(feature_name: str) -> str:
    """Get human-readable description of feature."""
    descriptions = {
        'EXT_SOURCE_MEAN': 'Average of external credit bureau scores',
        'EXT_SOURCE_1': 'Normalized score from external data source 1',
        'EXT_SOURCE_2': 'Normalized score from external data source 2',
        'EXT_SOURCE_3': 'Normalized score from external data source 3',
        'TRAJECTORY_SLOPE': 'Trend of payment behavior over time',
        'BEHAVIORAL_CONSISTENCY': 'Consistency of payments across all sources',
        'PAYMENT_DISCIPLINE_SCORE': 'Overall payment discipline rating',
        'DOCUMENT_COUNT': 'Number of documents submitted',
        'AMT_INCOME_TOTAL': 'Total annual income',
        'AMT_CREDIT': 'Loan amount requested',
        'AMT_ANNUITY': 'Loan annuity amount',
        'AGE_YEARS': 'Applicant age in years'
    }
    return descriptions.get(feature_name, feature_name.replace('_', ' ').title())

def calculate_feature_statistics(df: pd.DataFrame) -> Dict:
    """Calculate summary statistics for uploaded data."""
    return {
        'num_rows': len(df),
        'num_features': df.shape[1],
        'missing_values': df.isnull().sum().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }

def initialize_dice(model, df: pd.DataFrame, actionable_features: List[str]):
    """Initialize DiCE explainer object."""
    dice_df = df.copy()
    
    # Needs a target column logic for DiCE initialization
    preds = model.predict_proba(dice_df)[:, 1]
    dice_df['IS_APPROVED'] = (preds < 0.5).astype(int) 
    
    d = dice_ml.Data(dataframe=dice_df,
                     continuous_features=actionable_features, 
                     outcome_name='IS_APPROVED')
                     
    backend_model = ModelWrapper(model)
    m = dice_ml.Model(model=backend_model, backend="sklearn")
    
    exp = dice_ml.Dice(d, m, method="random")
    return exp
