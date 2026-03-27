"""
Financial Statement Fraud Detection System with Deep Learning and SHAP Explainability
======================================================================================

This Streamlit application implements a deep learning model for detecting financial 
statement/report fraud using the Kaggle Financial Reports Fraud Detection Dataset.

Features:
- Data preprocessing and exploration
- Deep Neural Network (DNN) model with TensorFlow/Keras
- SHAP (SHapley Additive exPlanations) for model interpretability
- Interactive visualizations and model performance metrics
- Real-time fraud prediction with explainability

Author: AI Assistant
Dataset: https://www.kaggle.com/datasets/ziya07/financial-reports-fraud-detection-data
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.impute import SimpleImputer

# SHAP for Explainability
import shap

# Data Processing
from data_processor import (
    load_and_process_data, 
    calculate_financial_ratios, 
    create_fraud_labels,
    prepare_model_features,
    process_full_dataset,
    FINANCIAL_ITEM_MAPPING
)

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Page configuration
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1565C0;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    .fraud-warning {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .safe-message {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class FinancialFraudDetector:
    """
    A comprehensive class for financial fraud detection using Deep Learning
    with SHAP explainability integration.
    """
    
    def __init__(self):
        """Initialize the fraud detection system."""
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        self.training_history = None
        self.shap_explainer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.raw_feature_names = []  # Original feature names before selection
        
    def load_real_data(self, data_path=None):
        """
        Load and process the financial dataset from local CSV file.
        
        Args:
            data_path: Path to the raw CSV file (default: financialreport.csv in same directory)
            
        Returns:
            Processed DataFrame with features and fraud labels
        """
        return process_full_dataset(data_path=data_path)
    
    def preprocess_data(self, df, target_column='Fraud'):
        """
        Preprocess the financial data for model training.
        
        Args:
            df: Input DataFrame with financial features
            target_column: Name of the target column
            
        Returns:
            Processed features and target arrays
        """
        # Separate features and target
        exclude_cols = ['company', 'period', 'Fraud', 'FraudProbability']
        
        if target_column in df.columns:
            # Get all numeric columns except excluded
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            X = df[feature_cols].copy()
            y = df[target_column].values
            
            # Store feature names
            self.raw_feature_names = feature_cols
        else:
            X = df
            y = None
            feature_cols = X.columns.tolist()
        
        # Handle inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=feature_cols)
        
        # Store feature names for SHAP
        self.feature_names = feature_cols
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def build_model(self, input_dim):
        """
        Build a deep neural network model for fraud detection.
        
        Architecture:
        - Input Layer
        - Multiple Dense layers with BatchNormalization and Dropout
        - Output layer with sigmoid activation for binary classification
        """
        model = models.Sequential([
            # Input Layer
            layers.Input(shape=(input_dim,)),
            
            # First Hidden Layer
            layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second Hidden Layer
            layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third Hidden Layer
            layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fourth Hidden Layer
            layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output Layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model with custom metrics
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                    epochs=100, batch_size=32, class_weight=None):
        """
        Train the deep learning model.
        """
        # Calculate class weights if not provided
        if class_weight is None:
            n_fraud = np.sum(y_train)
            n_non_fraud = len(y_train) - n_fraud
            class_weight = {0: 1.0, 1: min(n_non_fraud / n_fraud, 5.0)}
        
        # Define callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callback_list,
            verbose=1
        )
        
        self.is_trained = True
        return self.training_history
    
    def predict(self, X):
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet!")
        
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            # Reorder columns to match training
            X = X[self.feature_names]
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Get probabilities
        probabilities = self.model.predict(X_scaled, verbose=0).flatten()
        
        return probabilities
    
    def initialize_shap_explainer(self, X_background):
        """
        Initialize SHAP explainer for model interpretation.
        """
        # Use a subset of data as background for efficiency
        if len(X_background) > 100:
            background = shap.sample(X_background, 100)
        else:
            background = X_background
        
        # Create DeepExplainer for neural networks
        try:
            self.shap_explainer = shap.DeepExplainer(self.model, background)
        except Exception as e:
            st.warning(f"DeepExplainer failed, using KernelExplainer: {str(e)}")
            # Fallback to KernelExplainer
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict, background
            )
        
        return self.shap_explainer
    
    def get_shap_values(self, X, nsamples=100):
        """
        Calculate SHAP values for given data.
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized!")
        
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Calculate SHAP values
        if isinstance(self.shap_explainer, shap.DeepExplainer):
            shap_values = self.shap_explainer.shap_values(X_scaled)
        else:
            shap_values = self.shap_explainer.shap_values(X_scaled, nsamples=nsamples)
        
        return shap_values


def plot_training_history(history):
    """Plot training history metrics."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Loss', 'Model Accuracy', 'Model AUC', 'Precision/Recall'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(y=history.history['loss'], name='Training Loss', line=dict(color='#1E88E5')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_loss'], name='Validation Loss', line=dict(color='#FF5722')),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(y=history.history['accuracy'], name='Training Accuracy', line=dict(color='#1E88E5')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy', line=dict(color='#FF5722')),
        row=1, col=2
    )
    
    # AUC plot
    fig.add_trace(
        go.Scatter(y=history.history['auc'], name='Training AUC', line=dict(color='#1E88E5')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_auc'], name='Validation AUC', line=dict(color='#FF5722')),
        row=2, col=1
    )
    
    # Precision/Recall plot
    if 'precision' in history.history:
        fig.add_trace(
            go.Scatter(y=history.history['precision'], name='Training Precision', line=dict(color='#1E88E5')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(y=history.history['val_precision'], name='Validation Precision', line=dict(color='#FF5722')),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=True, title_text="Training History")
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Value")
    
    return fig


def plot_confusion_matrix_plotly(y_true, y_pred):
    """Create an interactive confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Non-Fraud', 'Predicted Fraud'],
        y=['Actual Non-Fraud', 'Actual Fraud'],
        hoverongaps=False,
        colorscale='Blues',
        showscale=True
    ))
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i, j]),
                font=dict(size=20, color='white' if cm[i, j] > cm.max()/2 else 'black'),
                showarrow=False
            )
    
    fig.update_layout(
        title='Confusion Matrix',
        width=500,
        height=500,
        xaxis_title='Predicted Label',
        yaxis_title='True Label'
    )
    
    return fig


def plot_roc_curve_plotly(y_true, y_prob):
    """Create an interactive ROC curve plot."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='#1E88E5', width=2),
        fill='tozeroy',
        fillcolor='rgba(30, 136, 229, 0.2)'
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        line=dict(color='#FF5722', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        width=600,
        height=500
    )
    
    return fig


def plot_shap_summary(shap_values, feature_names, max_display=20):
    """Create SHAP summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Ensure shap_values is a 2D numpy array
    shap_values = np.array(shap_values)
    
    # Handle 3D arrays (sometimes returned by DeepExplainer)
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze()
    
    # Ensure 2D shape
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    
    shap.summary_plot(
        shap_values, 
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=(10, 8)
    )
    
    plt.tight_layout()
    return fig


def plot_shap_bar(shap_values, feature_names, max_display=20):
    """Create SHAP bar plot for feature importance."""
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Convert to numpy array
    shap_values = np.array(shap_values)
    
    # Handle 3D arrays (sometimes returned by DeepExplainer for binary classification)
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze()
    
    # Ensure 2D shape
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Ensure mean_shap is 1D
    mean_shap = np.array(mean_shap).flatten()
    
    # Limit max_display to number of features
    max_display = min(max_display, len(feature_names), len(mean_shap))
    
    # Sort features - convert to list of integers for proper indexing
    sorted_idx = np.argsort(mean_shap)[-max_display:]
    sorted_idx = [int(i) for i in sorted_idx]  # Convert to list of integers
    
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = [float(mean_shap[i]) for i in sorted_idx]
    
    # Create bar plot
    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=sorted_features,
        orientation='h',
        marker=dict(color=sorted_values, colorscale='Blues')
    ))
    
    fig.update_layout(
        title='Feature Importance (Mean |SHAP Value|)',
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='Features',
        height=600,
        showlegend=False
    )
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Sidebar
    st.sidebar.markdown("<h1 style='text-align: center; color: #1E88E5;'>💰 Fraud Detection</h1>", 
                        unsafe_allow_html=True)
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Data Explorer", "🤖 Model Training", 
         "🔍 SHAP Explainability", "🎯 Prediction"],
        index=0
    )
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = FinancialFraudDetector()
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    detector = st.session_state.detector
    
    # Home Page
    if page == "🏠 Home":
        st.markdown("<h1 class='main-header'>Financial Statement Fraud Detection System</h1>", 
                    unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h3>Deep Learning + SHAP Explainability</h3>
            <p>Detect financial statement fraud using advanced neural networks with interpretable AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key features
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <h3>🧠</h3>
                <h4>Deep Learning</h4>
                <p>Multi-layer neural network with advanced regularization</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <h3>📊</h3>
                <h4>SHAP XAI</h4>
                <p>Explainable AI for transparent predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <h3>📈</h3>
                <h4>Real Data</h4>
                <p>Using actual Kaggle financial dataset</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='metric-card' style='text-align: center;'>
                <h3>🎯</h3>
                <h4>217 Features</h4>
                <p>Comprehensive financial indicators</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset Information
        st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
        
        st.info("""
        **Kaggle Dataset:** [Financial Reports Fraud Detection Data](https://www.kaggle.com/datasets/ziya07/financial-reports-fraud-detection-data)
        
        **Dataset Statistics:**
        - **999 samples** (company-period combinations)
        - **15 companies** with historical financial data
        - **217 features** including calculated financial ratios
        - **~18% fraud rate** based on financial anomaly detection
        
        **Feature Categories:**
        - Balance Sheet Items (Assets, Liabilities, Equity)
        - Cash Flow Items (Operating, Investing, Financing)
        - Profit Statement Items (Revenue, Expenses, Profit)
        - Financial Ratios (Profitability, Liquidity, Leverage, Efficiency)
        - Risk Scores (Altman Z-Score, Beneish M-Score)
        """)
        
        # Model Architecture
        st.markdown("<h2 class='sub-header'>Model Architecture</h2>", unsafe_allow_html=True)
        
        arch_col1, arch_col2 = st.columns([2, 1])
        
        with arch_col1:
            st.code("""
Deep Neural Network Architecture:
┌─────────────────────────────────────┐
│  Input Layer (217 Features)         │
├─────────────────────────────────────┤
│  Dense (256) + BatchNorm + Dropout  │
├─────────────────────────────────────┤
│  Dense (128) + BatchNorm + Dropout  │
├─────────────────────────────────────┤
│  Dense (64) + BatchNorm + Dropout   │
├─────────────────────────────────────┤
│  Dense (32) + BatchNorm + Dropout   │
├─────────────────────────────────────┤
│  Output (Sigmoid) - Fraud/Non-Fraud│
└─────────────────────────────────────┘

Features:
• Adam Optimizer with adaptive learning rate
• Binary Cross-Entropy Loss
• Class Weighting for imbalanced data
• Early Stopping & Learning Rate Reduction
            """, language='text')
        
        with arch_col2:
            st.markdown("""
            **Performance Metrics:**
            - Accuracy
            - Precision
            - Recall
            - F1-Score
            - AUC-ROC
            
            **Explainability:**
            - SHAP DeepExplainer
            - Feature Importance
            - Individual Prediction Explanations
            """)
    
    # Data Explorer Page
    elif page == "📊 Data Explorer":
        st.markdown("<h1 class='main-header'>📊 Data Explorer</h1>", unsafe_allow_html=True)
        
        # Data source selection
        st.markdown("<h2 class='sub-header'>Data Source</h2>", unsafe_allow_html=True)
        
        data_source = st.radio(
            "Select data source:",
            ["Load Local Dataset", "Upload Custom Dataset"],
            horizontal=True
        )
        
        if data_source == "Load Local Dataset":
            if st.button("Load Financial Dataset", type="primary"):
                with st.spinner("Loading and processing financial dataset..."):
                    try:
                        df = detector.load_real_data()
                        st.session_state.data = df
                        st.success(f"Loaded {len(df)} samples with {len(df.columns)} features")
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
        
        else:
            uploaded_file = st.file_uploader(
                "Upload Custom Dataset (CSV)",
                type=['csv'],
                help="Upload a processed financial dataset with features"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.data = df
                    st.success(f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        # Display data
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Data Overview
            st.markdown("<h2 class='sub-header'>Data Overview</h2>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(df))
            
            with col2:
                feature_count = len([c for c in df.columns if c not in ['company', 'period', 'Fraud', 'FraudProbability']])
                st.metric("Features", feature_count)
            
            with col3:
                if 'Fraud' in df.columns:
                    st.metric("Fraud Cases", df['Fraud'].sum())
                else:
                    st.metric("Fraud Cases", "N/A")
            
            with col4:
                if 'Fraud' in df.columns:
                    st.metric("Fraud Rate", f"{df['Fraud'].mean()*100:.1f}%")
                else:
                    st.metric("Fraud Rate", "N/A")
            
            # Company distribution
            if 'company' in df.columns:
                st.markdown("<h3>Samples per Company</h3>", unsafe_allow_html=True)
                company_counts = df['company'].value_counts()
                fig = px.bar(x=company_counts.index.astype(str), y=company_counts.values,
                           labels={'x': 'Company', 'y': 'Sample Count'},
                           title='Number of Samples per Company')
                st.plotly_chart(fig, use_container_width=True)
            
            # Data Preview
            st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
            # Show key columns
            key_cols = ['company', 'period', 'Fraud', 'FraudProbability']
            display_cols = [c for c in key_cols if c in df.columns]
            display_cols += [c for c in df.columns[:15] if c not in display_cols]
            st.dataframe(df[display_cols].head(10), use_container_width=True)
            
            # Key Financial Ratios Distribution
            st.markdown("<h2 class='sub-header'>Key Financial Indicators</h2>", unsafe_allow_html=True)
            
            ratio_cols = ['ROA', 'ROE', 'CurrentRatio', 'DebtToEquity', 'GrossProfitMargin', 
                         'NetProfitMargin', 'AltmanZScore', 'BeneishMScore', 'AccrualsRatio']
            available_ratios = [c for c in ratio_cols if c in df.columns]
            
            selected_ratios = st.multiselect(
                "Select financial indicators to visualize:",
                available_ratios,
                default=available_ratios[:4]
            )
            
            if selected_ratios and 'Fraud' in df.columns:
                fig = make_subplots(
                    rows=len(selected_ratios), cols=1,
                    subplot_titles=selected_ratios,
                    vertical_spacing=0.05
                )
                
                for i, col in enumerate(selected_ratios):
                    fig.add_trace(
                        go.Histogram(
                            x=df[df['Fraud']==0][col].dropna(),
                            name=f'{col} (Non-Fraud)',
                            marker_color='#1E88E5',
                            opacity=0.7,
                            showlegend=(i==0)
                        ),
                        row=i+1, col=1
                    )
                    fig.add_trace(
                        go.Histogram(
                            x=df[df['Fraud']==1][col].dropna(),
                            name=f'{col} (Fraud)',
                            marker_color='#FF5722',
                            opacity=0.7,
                            showlegend=(i==0)
                        ),
                        row=i+1, col=1
                    )
                
                fig.update_layout(height=200*len(selected_ratios), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation with Fraud
            if 'Fraud' in df.columns:
                st.markdown("<h3>Feature Correlation with Fraud</h3>", unsafe_allow_html=True)
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                exclude_cols = ['company', 'period', 'Fraud', 'FraudProbability']
                feature_cols = [c for c in numeric_cols if c not in exclude_cols]
                
                correlations = df[feature_cols + ['Fraud']].corr()['Fraud'].drop('Fraud')
                top_corr = correlations.abs().sort_values(ascending=False).head(20)
                
                fig = go.Figure(go.Bar(
                    x=top_corr.values,
                    y=top_corr.index,
                    orientation='h',
                    marker=dict(color=top_corr.values, colorscale='RdBu')
                ))
                
                fig.update_layout(
                    title='Top 20 Features by Correlation with Fraud',
                    xaxis_title='Absolute Correlation',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Model Training Page
    elif page == "🤖 Model Training":
        st.markdown("<h1 class='main-header'>🤖 Model Training</h1>", unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.warning("Please load data in the Data Explorer page first!")
        else:
            df = st.session_state.data
            
            if 'Fraud' not in df.columns:
                st.error("The dataset must contain a 'Fraud' target column!")
            else:
                # Training Parameters
                st.markdown("<h2 class='sub-header'>Training Parameters</h2>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    test_size = st.slider("Test Set Size (%):", 10, 40, 20)
                    epochs = st.slider("Max Epochs:", 10, 200, 100)
                
                with col2:
                    batch_size = st.select_slider(
                        "Batch Size:",
                        options=[16, 32, 64, 128, 256],
                        value=32
                    )
                    learning_rate = st.select_slider(
                        "Learning Rate:",
                        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                        value=0.001
                    )
                
                with col3:
                    validation_split = st.slider("Validation Split (%):", 5, 30, 15)
                    random_state = st.number_input("Random State:", value=42)
                
                # Advanced Options
                with st.expander("🔧 Advanced Options"):
                    use_class_weights = st.checkbox("Use Class Weights", value=True)
                    early_stopping_patience = st.slider("Early Stopping Patience:", 5, 30, 15)
                
                # Start Training
                if st.button("🚀 Start Training", type="primary"):
                    with st.spinner("Training model... This may take a few minutes."):
                        # Preprocess data
                        X, y = detector.preprocess_data(df, 'Fraud')
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=random_state,
                            stratify=y
                        )
                        
                        # Further split for validation
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train, y_train, test_size=validation_split/100,
                            random_state=random_state, stratify=y_train
                        )
                        
                        # Store for later use
                        detector.X_train = X_train
                        detector.X_test = X_test
                        detector.y_train = y_train
                        detector.y_test = y_test
                        
                        # Build model
                        input_dim = X_train.shape[1]
                        detector.build_model(input_dim)
                        
                        # Update learning rate
                        detector.model.optimizer.learning_rate.assign(learning_rate)
                        
                        # Display model summary
                        st.markdown("<h3>Model Architecture</h3>", unsafe_allow_html=True)
                        model_summary = []
                        detector.model.summary(print_fn=lambda x: model_summary.append(x))
                        st.code('\n'.join(model_summary), language='text')
                        
                        # Train model
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        history = detector.train_model(
                            X_train, y_train,
                            X_val=X_val, y_val=y_val,
                            epochs=epochs,
                            batch_size=batch_size,
                            class_weight={0: 1.0, 1: 3.0} if use_class_weights else None
                        )
                        
                        progress_bar.progress(1.0)
                        st.success("Training completed!")
                        
                        # Plot training history
                        st.markdown("<h2 class='sub-header'>Training History</h2>", unsafe_allow_html=True)
                        st.plotly_chart(plot_training_history(history), use_container_width=True)
                        
                        # Evaluate on test set
                        st.markdown("<h2 class='sub-header'>Model Evaluation</h2>", unsafe_allow_html=True)
                        
                        y_pred_prob = detector.model.predict(X_test, verbose=0).flatten()
                        y_pred = (y_pred_prob > 0.5).astype(int)
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                        with col2:
                            st.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
                        with col3:
                            st.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
                        with col4:
                            st.metric("F1-Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
                        
                        # Confusion Matrix
                        st.plotly_chart(plot_confusion_matrix_plotly(y_test, y_pred), use_container_width=True)
                        
                        # ROC Curve
                        st.plotly_chart(plot_roc_curve_plotly(y_test, y_pred_prob), use_container_width=True)
                        
                        # Classification Report
                        st.markdown("<h3>Classification Report</h3>", unsafe_allow_html=True)
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                        
                        st.session_state.model_trained = True
                        
                        # Initialize SHAP explainer
                        with st.spinner("Initializing SHAP explainer..."):
                            detector.initialize_shap_explainer(X_train[:200])
                        st.success("SHAP explainer initialized!")
    
    # SHAP Explainability Page
    elif page == "🔍 SHAP Explainability":
        st.markdown("<h1 class='main-header'>🔍 SHAP Explainability</h1>", unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first in the Model Training page!")
        else:
            st.markdown("""
            <div style='margin-bottom: 2rem;'>
                <h3>Understanding Model Predictions with SHAP</h3>
                <p>SHAP (SHapley Additive exPlanations) provides interpretable explanations for model predictions,
                showing how each feature contributes to the final fraud prediction.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # SHAP Analysis Options
            st.markdown("<h2 class='sub-header'>SHAP Analysis</h2>", unsafe_allow_html=True)
            
            analysis_type = st.radio(
                "Select Analysis Type:",
                ["Feature Importance (Global)", "Individual Prediction Explanation"],
                horizontal=True
            )
            
            if analysis_type == "Feature Importance (Global)":
                st.markdown("""
                **Global Feature Importance**: Shows which features are most influential in the model's fraud detection decisions across all predictions.
                """)
                
                n_samples = st.slider("Number of samples for SHAP analysis:", 50, 300, 100)
                
                if st.button("Generate SHAP Analysis", type="primary"):
                    with st.spinner("Calculating SHAP values... This may take a moment."):
                        # Get sample data
                        X_sample = pd.DataFrame(
                            detector.X_test[:n_samples], 
                            columns=detector.feature_names
                        )
                        
                        # Calculate SHAP values
                        shap_values = detector.get_shap_values(X_sample)
                        
                        # Feature Importance Bar Plot
                        st.markdown("<h3>Feature Importance (Mean |SHAP Value|)</h3>", unsafe_allow_html=True)
                        st.plotly_chart(plot_shap_bar(shap_values, detector.feature_names), use_container_width=True)
                        
                        # SHAP Summary Plot
                        st.markdown("<h3>SHAP Summary Plot</h3>", unsafe_allow_html=True)
                        st.markdown("""
                        This plot shows the distribution of SHAP values for each feature. 
                        Red indicates high feature values, blue indicates low values.
                        """)
                        
                        fig = plot_shap_summary(shap_values, detector.feature_names)
                        st.pyplot(fig)
                        plt.clf()
                        
                        # Detailed SHAP values
                        st.markdown("<h3>Top Features Analysis</h3>", unsafe_allow_html=True)
                        
                        if isinstance(shap_values, list):
                            sv = shap_values[0]
                        else:
                            sv = shap_values
                        
                        # Ensure proper array format
                        sv = np.array(sv)
                        if sv.ndim == 3:
                            sv = sv.squeeze()
                        
                        mean_shap = np.abs(sv).mean(axis=0)
                        mean_shap = np.array(mean_shap).flatten()
                        
                        top_features = pd.DataFrame({
                            'Feature': detector.feature_names,
                            'Mean |SHAP|': mean_shap
                        }).sort_values('Mean |SHAP|', ascending=False).head(20)
                        
                        st.dataframe(top_features, use_container_width=True)
            
            elif analysis_type == "Individual Prediction Explanation":
                st.markdown("""
                **Individual Prediction Explanation**: Shows how each feature contributes to a specific prediction.
                """)
                
                # Select a sample
                sample_idx = st.slider("Select sample index:", 0, len(detector.X_test)-1, 0)
                
                if st.button("Explain Prediction", type="primary"):
                    with st.spinner("Calculating SHAP values for individual prediction..."):
                        # Get single sample
                        X_single = pd.DataFrame(
                            [detector.X_test[sample_idx]], 
                            columns=detector.feature_names
                        )
                        
                        # Get prediction
                        prob = detector.predict(X_single)[0]
                        prediction = "FRAUD" if prob > 0.5 else "NON-FRAUD"
                        
                        # Display prediction
                        if prediction == "FRAUD":
                            st.markdown(f"""
                            <div class='fraud-warning'>
                                <h3>⚠️ Prediction: {prediction}</h3>
                                <p>Fraud Probability: {prob:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='safe-message'>
                                <h3>✅ Prediction: {prediction}</h3>
                                <p>Fraud Probability: {prob:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Calculate SHAP values
                        shap_values = detector.get_shap_values(X_single)
                        
                        # Handle different SHAP value formats
                        if isinstance(shap_values, list):
                            sv = np.array(shap_values[0])
                        else:
                            sv = np.array(shap_values)
                        
                        # Handle 3D arrays
                        if sv.ndim == 3:
                            sv = sv.squeeze()
                        
                        # Get the first (and only) sample's SHAP values
                        if sv.ndim == 2:
                            sv = sv[0]
                        
                        # Ensure 1D
                        sv = sv.flatten()
                        
                        # Create waterfall plot data
                        feature_contributions = pd.DataFrame({
                            'Feature': detector.feature_names,
                            'SHAP Value': sv,
                            'Feature Value': X_single.iloc[0].values
                        }).sort_values('SHAP Value', key=abs, ascending=False)
                        
                        # Top contributors
                        st.markdown("<h3>Top Contributing Features</h3>", unsafe_allow_html=True)
                        
                        fig = go.Figure()
                        
                        top_n = 15
                        top_features = feature_contributions.head(top_n)
                        
                        colors = ['#FF5722' if x > 0 else '#1E88E5' for x in top_features['SHAP Value']]
                        
                        fig.add_trace(go.Bar(
                            x=top_features['SHAP Value'],
                            y=top_features['Feature'],
                            orientation='h',
                            marker_color=colors,
                            text=[f'{v:.3f}' for v in top_features['SHAP Value']],
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            title='Feature Contributions to Prediction',
                            xaxis_title='SHAP Value (Impact on Fraud Probability)',
                            yaxis_title='Feature',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed breakdown
                        st.markdown("<h3>Feature Details</h3>", unsafe_allow_html=True)
                        st.dataframe(feature_contributions.head(20), use_container_width=True)
    
    # Prediction Page
    elif page == "🎯 Prediction":
        st.markdown("<h1 class='main-header'>🎯 Fraud Prediction</h1>", unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first in the Model Training page!")
        else:
            prediction_mode = st.radio(
                "Select Prediction Mode:",
                ["Single Prediction", "Batch Prediction"],
                horizontal=True
            )
            
            if prediction_mode == "Single Prediction":
                st.markdown("""
                <div style='margin-bottom: 1rem;'>
                    <p>Enter the feature values for fraud prediction. 
                    The model will analyze the financial indicators and provide a fraud risk assessment.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create input form with key features
                st.markdown("<h3>Key Financial Indicators</h3>", unsafe_allow_html=True)
                
                input_data = {}
                
                # Use most important features based on correlation
                key_features = ['ROA', 'ROE', 'CurrentRatio', 'DebtToEquity', 'GrossProfitMargin',
                               'NetProfitMargin', 'AltmanZScore', 'BeneishMScore', 'AccrualsRatio',
                               'AssetTurnover', 'OperatingCashFlowToDebt', 'QuickRatio', 'DebtRatio']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h4>Profitability</h4>", unsafe_allow_html=True)
                    input_data['ROA'] = st.number_input("ROA (Return on Assets)", value=0.06, format="%.4f")
                    input_data['ROE'] = st.number_input("ROE (Return on Equity)", value=0.12, format="%.4f")
                    input_data['GrossProfitMargin'] = st.number_input("Gross Profit Margin", value=0.35, format="%.4f")
                    input_data['NetProfitMargin'] = st.number_input("Net Profit Margin", value=0.08, format="%.4f")
                    
                    st.markdown("<h4>Liquidity</h4>", unsafe_allow_html=True)
                    input_data['CurrentRatio'] = st.number_input("Current Ratio", value=1.8, format="%.2f")
                    input_data['QuickRatio'] = st.number_input("Quick Ratio", value=1.2, format="%.2f")
                
                with col2:
                    st.markdown("<h4>Leverage</h4>", unsafe_allow_html=True)
                    input_data['DebtToEquity'] = st.number_input("Debt to Equity", value=1.5, format="%.2f")
                    input_data['DebtRatio'] = st.number_input("Debt Ratio", value=0.6, format="%.4f")
                    
                    st.markdown("<h4>Risk Scores</h4>", unsafe_allow_html=True)
                    input_data['AltmanZScore'] = st.number_input("Altman Z-Score", value=3.0, format="%.2f")
                    input_data['BeneishMScore'] = st.number_input("Beneish M-Score", value=-1.5, format="%.2f")
                    input_data['AccrualsRatio'] = st.number_input("Accruals Ratio", value=0.02, format="%.4f")
                    
                    st.markdown("<h4>Efficiency</h4>", unsafe_allow_html=True)
                    input_data['AssetTurnover'] = st.number_input("Asset Turnover", value=0.8, format="%.2f")
                    input_data['OperatingCashFlowToDebt'] = st.number_input("Operating Cash Flow to Debt", value=0.3, format="%.2f")
                
                # Predict button
                if st.button("🔍 Predict Fraud Risk", type="primary"):
                    with st.spinner("Analyzing..."):
                        # Create DataFrame with all features (fill missing with median)
                        input_df = pd.DataFrame([input_data])
                        
                        # Add all features that model expects
                        for feat in detector.feature_names:
                            if feat not in input_df.columns:
                                input_df[feat] = 0.0
                        
                        # Reorder columns
                        input_df = input_df[detector.feature_names]
                        
                        # Predict
                        fraud_prob = detector.predict(input_df)[0]
                        
                        # Display result
                        st.markdown("<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
                        
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=fraud_prob * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Fraud Probability"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkred" if fraud_prob > 0.5 else "darkgreen"},
                                'steps': [
                                    {'range': [0, 30], 'color': "#E8F5E9"},
                                    {'range': [30, 50], 'color': "#FFF3E0"},
                                    {'range': [50, 70], 'color': "#FFEBEE"},
                                    {'range': [70, 100], 'color': "#FFCDD2"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk assessment
                        if fraud_prob > 0.7:
                            st.markdown(f"""
                            <div class='fraud-warning'>
                                <h3>⚠️ HIGH RISK - FRAUD SUSPECTED</h3>
                                <p>Fraud Probability: <strong>{fraud_prob:.2%}</strong></p>
                                <p>This financial statement shows significant indicators of potential fraud.
                                Immediate investigation is recommended.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif fraud_prob > 0.5:
                            st.markdown(f"""
                            <div class='fraud-warning'>
                                <h3>⚠️ MODERATE RISK - REVIEW REQUIRED</h3>
                                <p>Fraud Probability: <strong>{fraud_prob:.2%}</strong></p>
                                <p>This financial statement shows some concerning indicators.
                                Additional review is recommended.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif fraud_prob > 0.3:
                            st.markdown(f"""
                            <div class='safe-message'>
                                <h3>⚡ LOW-MEDIUM RISK</h3>
                                <p>Fraud Probability: <strong>{fraud_prob:.2%}</strong></p>
                                <p>This financial statement shows minor risk indicators.
                                Standard monitoring recommended.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='safe-message'>
                                <h3>✅ LOW RISK - APPEARS LEGITIMATE</h3>
                                <p>Fraud Probability: <strong>{fraud_prob:.2%}</strong></p>
                                <p>This financial statement appears to be legitimate.
                                No immediate concerns detected.</p>
                            </div>
                            """, unsafe_allow_html=True)
            
            else:  # Batch Prediction
                st.markdown("<h2 class='sub-header'>Batch Prediction</h2>", unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader(
                    "Upload data file for batch prediction (CSV)",
                    type=['csv']
                )
                
                if uploaded_file is not None:
                    try:
                        batch_df = pd.read_csv(uploaded_file)
                        st.dataframe(batch_df.head(), use_container_width=True)
                        
                        if st.button("Run Batch Prediction", type="primary"):
                            with st.spinner("Processing batch predictions..."):
                                # Ensure all features are present
                                for feat in detector.feature_names:
                                    if feat not in batch_df.columns:
                                        batch_df[feat] = 0.0
                                
                                # Reorder columns
                                X_batch = batch_df[detector.feature_names]
                                
                                # Predict
                                probabilities = detector.predict(X_batch)
                                
                                # Add predictions to dataframe
                                batch_df['Fraud_Probability'] = probabilities
                                batch_df['Prediction'] = (probabilities > 0.5).map({True: 'Fraud', False: 'Non-Fraud'})
                                batch_df['Risk_Level'] = pd.cut(
                                    probabilities,
                                    bins=[0, 0.3, 0.5, 0.7, 1.0],
                                    labels=['Low', 'Low-Medium', 'Medium-High', 'High']
                                )
                                
                                # Display results
                                st.markdown("<h3>Batch Prediction Results</h3>", unsafe_allow_html=True)
                                st.dataframe(batch_df, use_container_width=True)
                                
                                # Summary statistics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total Records", len(batch_df))
                                with col2:
                                    st.metric("Predicted Fraud", (batch_df['Prediction'] == 'Fraud').sum())
                                with col3:
                                    st.metric("Average Fraud Probability", f"{probabilities.mean():.2%}")
                                
                                # Distribution plot
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(
                                    x=probabilities,
                                    nbinsx=20,
                                    marker_color='#1E88E5',
                                    opacity=0.7
                                ))
                                fig.update_layout(
                                    title='Distribution of Fraud Probabilities',
                                    xaxis_title='Fraud Probability',
                                    yaxis_title='Count'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Download results
                                csv = batch_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download Predictions (CSV)",
                                    data=csv,
                                    file_name="fraud_predictions.csv",
                                    mime="text/csv"
                                )
                    
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; font-size: 0.8rem;'>
        <p>Financial Fraud Detection System</p>
        <p>Deep Learning + SHAP XAI</p>
        <p>Kaggle Dataset Integration</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
