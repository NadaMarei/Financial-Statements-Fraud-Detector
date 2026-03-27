"""
Financial Fraud Detection Utilities
===================================

This module provides utility functions for data processing,
feature engineering, and financial analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


def calculate_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional financial ratios from raw financial data.
    
    Args:
        df: DataFrame with basic financial metrics
        
    Returns:
        DataFrame with additional calculated ratios
    """
    df = df.copy()
    
    # Profitability Ratios
    if all(col in df.columns for col in ['Revenue', 'Gross_Profit']):
        df['Gross_Margin'] = df['Gross_Profit'] / df['Revenue']
    
    if all(col in df.columns for col in ['Revenue', 'Operating_Income']):
        df['Operating_Margin'] = df['Operating_Income'] / df['Revenue']
    
    if all(col in df.columns for col in ['Revenue', 'Net_Income']):
        df['Net_Margin'] = df['Net_Income'] / df['Revenue']
    
    # Liquidity Ratios
    if all(col in df.columns for col in ['Current_Assets', 'Current_Liabilities']):
        df['Current_Ratio'] = df['Current_Assets'] / df['Current_Liabilities']
    
    if all(col in df.columns for col in ['Current_Assets', 'Inventory', 'Current_Liabilities']):
        df['Quick_Ratio'] = (df['Current_Assets'] - df['Inventory']) / df['Current_Liabilities']
    
    # Leverage Ratios
    if all(col in df.columns for col in ['Total_Liabilities', 'Total_Equity']):
        df['Debt_to_Equity'] = df['Total_Liabilities'] / df['Total_Equity']
    
    if all(col in df.columns for col in ['Total_Debt', 'Total_Assets']):
        df['Debt_Ratio'] = df['Total_Debt'] / df['Total_Assets']
    
    # Efficiency Ratios
    if all(col in df.columns for col in ['Revenue', 'Total_Assets']):
        df['Asset_Turnover'] = df['Revenue'] / df['Total_Assets']
    
    if all(col in df.columns for col in ['Cost_of_Goods_Sold', 'Inventory']):
        df['Inventory_Turnover'] = df['Cost_of_Goods_Sold'] / df['Inventory']
    
    # Return Ratios
    if all(col in df.columns for col in ['Net_Income', 'Total_Assets']):
        df['ROA'] = df['Net_Income'] / df['Total_Assets']
    
    if all(col in df.columns for col in ['Net_Income', 'Total_Equity']):
        df['ROE'] = df['Net_Income'] / df['Total_Equity']
    
    return df


def calculate_altman_z_score(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate Altman Z-Score for bankruptcy prediction.
    
    Z-Score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    
    Where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Equity / Total Liabilities
    X5 = Sales / Total Assets
    
    Interpretation:
    Z > 2.99: Safe Zone
    1.81 < Z < 2.99: Grey Zone
    Z < 1.81: Distress Zone
    """
    z_scores = np.zeros(len(df))
    
    try:
        # X1: Working Capital / Total Assets
        x1 = (df.get('Current_Assets', 0) - df.get('Current_Liabilities', 0)) / df['Total_Assets']
        
        # X2: Retained Earnings / Total Assets
        x2 = df.get('Retained_Earnings', df.get('Total_Equity', 0) * 0.5) / df['Total_Assets']
        
        # X3: EBIT / Total Assets
        x3 = df.get('EBIT', df.get('Operating_Income', df.get('Net_Income', 0))) / df['Total_Assets']
        
        # X4: Market Equity / Total Liabilities
        x4 = df.get('Market_Cap', df.get('Total_Equity', 0)) / df['Total_Liabilities']
        
        # X5: Sales / Total Assets
        x5 = df.get('Revenue', df.get('Sales', 0)) / df['Total_Assets']
        
        # Calculate Z-Score
        z_scores = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        
        # Handle inf and nan
        z_scores = np.nan_to_num(z_scores, nan=3.0, posinf=5.0, neginf=-1.0)
        
    except Exception as e:
        print(f"Error calculating Z-Score: {e}")
        z_scores = np.full(len(df), 3.0)
    
    return z_scores


def calculate_beneish_m_score(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate Beneish M-Score for earnings manipulation detection.
    
    M-Score = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI 
              + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    
    Interpretation:
    M < -2.22: Non-manipulator
    M > -2.22: Potential manipulator
    """
    m_scores = np.zeros(len(df))
    
    try:
        # DSRI: Days Sales Receivable Index
        dsri = df.get('DSRI', df.get('Days_Sales_Outstanding', 45) / 45)
        
        # GMI: Gross Margin Index
        gmi = df.get('GMI', 1 - df.get('Gross_Profit_Margin', 0.35))
        
        # AQI: Asset Quality Index
        aqi = df.get('AQI', 1.0)
        
        # SGI: Sales Growth Index
        sgi = 1 + df.get('Revenue_Growth', 0.05)
        
        # DEPI: Depreciation Index
        depi = df.get('DEPI', 1.0)
        
        # SGAI: Sales, General and Administrative Index
        sgai = df.get('SGAI', 1.0)
        
        # TATA: Total Accruals to Total Assets
        tata = df.get('Accruals_Ratio', 0.02)
        
        # LVGI: Leverage Index
        lvgi = df.get('Debt_to_Equity', 1.0) / 1.0
        
        # Calculate M-Score
        m_scores = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 
                    0.892 * sgi + 0.115 * depi - 0.172 * sgai + 
                    4.679 * tata - 0.327 * lvgi)
        
        # Handle inf and nan
        m_scores = np.nan_to_num(m_scores, nan=-1.5, posinf=2.0, neginf=-3.0)
        
    except Exception as e:
        print(f"Error calculating M-Score: {e}")
        m_scores = np.full(len(df), -1.5)
    
    return m_scores


def detect_anomalies(df: pd.DataFrame, columns: List[str], 
                     threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalies using Z-score method.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for anomalies
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with anomaly flags
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f'{col}_anomaly'] = np.abs((df[col] - mean) / std) > threshold
            else:
                df[f'{col}_anomaly'] = False
    
    return df


def impute_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Impute missing values in the DataFrame.
    
    Args:
        df: Input DataFrame with potential missing values
        strategy: Imputation strategy ('mean', 'median', 'most_frequent')
        
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    imputer = SimpleImputer(strategy=strategy)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df


def create_interaction_features(df: pd.DataFrame, 
                               feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features between pairs of features.
    
    Args:
        df: Input DataFrame
        feature_pairs: List of feature name tuples to create interactions
        
    Returns:
        DataFrame with interaction features
    """
    df = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            # Multiplication interaction
            df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            
            # Division interaction (with safe division)
            with np.errstate(divide='ignore', invalid='ignore'):
                df[f'{feat1}_div_{feat2}'] = np.where(
                    df[feat2] != 0,
                    df[feat1] / df[feat2],
                    0
                )
    
    return df


def normalize_features(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, object]:
    """
    Normalize features using specified method.
    
    Args:
        X: Feature matrix
        method: Normalization method ('standard' or 'minmax')
        
    Returns:
        Normalized features and fitted scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                               y_prob: np.ndarray) -> Dict:
    """
    Calculate comprehensive model performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        
    Returns:
        Dictionary of performance metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, matthews_corrcoef
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    return metrics


def generate_risk_report(fraud_probability: float, 
                         shap_values: np.ndarray,
                         feature_names: List[str],
                         feature_values: np.ndarray) -> str:
    """
    Generate a human-readable risk assessment report.
    
    Args:
        fraud_probability: Predicted fraud probability
        shap_values: SHAP values for the prediction
        feature_names: List of feature names
        feature_values: Feature values for the prediction
        
    Returns:
        Formatted risk report string
    """
    # Determine risk level
    if fraud_probability > 0.7:
        risk_level = "HIGH"
        risk_emoji = "🔴"
    elif fraud_probability > 0.5:
        risk_level = "MEDIUM-HIGH"
        risk_emoji = "🟠"
    elif fraud_probability > 0.3:
        risk_level = "MEDIUM"
        risk_emoji = "🟡"
    else:
        risk_level = "LOW"
        risk_emoji = "🟢"
    
    # Get top contributing features
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-5:][::-1]
    
    report = f"""
{'='*50}
FINANCIAL FRAUD RISK ASSESSMENT REPORT
{'='*50}

Overall Risk Level: {risk_emoji} {risk_level}
Fraud Probability: {fraud_probability:.2%}

Top Contributing Factors:
{'-'*30}
"""
    
    for i, idx in enumerate(top_indices, 1):
        direction = "↑ Increases" if shap_values[idx] > 0 else "↓ Decreases"
        report += f"""
{i}. {feature_names[idx]}
   Value: {feature_values[idx]:.4f}
   Impact: {direction} fraud risk
   SHAP Value: {shap_values[idx]:.4f}
"""
    
    report += f"""
{'='*50}
Recommendation: 
"""
    
    if fraud_probability > 0.7:
        report += "Immediate investigation required. High probability of fraudulent activity."
    elif fraud_probability > 0.5:
        report += "Detailed review recommended. Several risk indicators present."
    elif fraud_probability > 0.3:
        report += "Monitor closely. Some risk indicators detected."
    else:
        report += "Standard monitoring. No significant risk indicators."
    
    report += f"\n{'='*50}"
    
    return report


def save_model_artifacts(model, scaler, feature_names: List[str], 
                        filepath: str = 'model_artifacts'):
    """
    Save model artifacts for later use.
    
    Args:
        model: Trained Keras model
        scaler: Fitted scaler
        feature_names: List of feature names
        filepath: Base filepath for saving artifacts
    """
    import os
    import pickle
    
    # Create directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
    
    # Save model
    model.save(os.path.join(filepath, 'model.h5'))
    
    # Save scaler
    with open(os.path.join(filepath, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open(os.path.join(filepath, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(f"Model artifacts saved to {filepath}")


def load_model_artifacts(filepath: str = 'model_artifacts'):
    """
    Load saved model artifacts.
    
    Args:
        filepath: Base filepath for loading artifacts
        
    Returns:
        Tuple of (model, scaler, feature_names)
    """
    import os
    import pickle
    from tensorflow import keras
    
    # Load model
    model = keras.models.load_model(os.path.join(filepath, 'model.h5'))
    
    # Load scaler
    with open(os.path.join(filepath, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature names
    with open(os.path.join(filepath, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, scaler, feature_names
