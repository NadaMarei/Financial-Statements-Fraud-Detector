# Financial Statement Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![SHAP](https://img.shields.io/badge/SHAP-0.42%2B-green)

**Deep Learning + Explainable AI for Financial Fraud Detection**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Dataset](#dataset) • [SHAP Explainability](#shap-explainability)

</div>

---

## Overview

This application implements a comprehensive financial statements fraud detection system using deep learning with SHAP (SHapley Additive exPlanations) for model interpretability. It's designed to detect fraudulent financial statements and reports using the **real Kaggle Financial Reports Fraud Detection Dataset**.

## Dataset

**Source**: [Kaggle - Financial Reports Fraud Detection Data](https://www.kaggle.com/datasets/ziya07/financial-reports-fraud-detection-data)

**Dataset Statistics**:
- **999 samples** (company-period combinations)
- **15 companies** with historical financial data
- **217 features** including:
  - Balance Sheet Items (Assets, Liabilities, Equity)
  - Cash Flow Items (Operating, Investing, Financing)
  - Profit Statement Items (Revenue, Expenses, Profit)
  - Financial Ratios (Profitability, Liquidity, Leverage, Efficiency)
  - Risk Scores (Altman Z-Score, Beneish M-Score)
- **~18% fraud rate** based on financial anomaly detection

## Features

### Core Capabilities
- **Deep Neural Network Model**: Multi-layer architecture with batch normalization and dropout for robust fraud detection
- **SHAP Explainability**: Full integration of SHAP for transparent and interpretable predictions
- **Interactive UI**: User-friendly Streamlit interface with multiple analysis pages
- **Real Kaggle Data**: Uses actual financial data from Chinese-listed companies

### Analysis Pages
1. **Home**: Overview of the system, model architecture, and quick start guide
2. **Data Explorer**: Load Kaggle data, explore financial indicators, view correlations
3. **Model Training**: Configure and train the deep learning model with customizable parameters
4. **SHAP Explainability**: Global feature importance and individual prediction explanations
5. **Prediction**: Single and batch predictions with risk assessment

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Quick Start

1. **Clone or download the repository**

2. **Create virtual environment and install dependencies**
```bash
cd financial_fraud_detection
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

## Usage

### 1. Data Loading

The system automatically loads and processes the Kaggle dataset:

1. Navigate to "Data Explorer"
2. Click "Load Kaggle Financial Dataset"
3. The dataset will be automatically:
   - Downloaded via kagglehub
   - Pivoted from long to wide format
   - Enhanced with 30+ calculated financial ratios
   - Labeled for fraud based on financial anomalies

### 2. Model Training

1. Go to "Model Training" page
2. Configure training parameters:
   - Test set size (default: 20%)
   - Maximum epochs (default: 100)
   - Batch size (default: 32)
   - Learning rate (default: 0.001)
3. Click "Start Training"
4. Review training history and model performance metrics

### 3. SHAP Analysis

After training, explore model explainability:

- **Global Feature Importance**: See which features are most influential overall
- **Individual Predictions**: Understand why a specific prediction was made

### 4. Making Predictions

#### Single Prediction
1. Go to "Prediction" page
2. Enter financial data values for key indicators
3. Click "Predict Fraud Risk"
4. View probability, risk level, and feature contributions

#### Batch Prediction
1. Prepare a CSV file with financial features
2. Upload the file
3. Click "Run Batch Prediction"
4. Download results

## Model Architecture

```
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
```

### Key Features:
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Regularization**: Dropout (0.2-0.3) and Batch Normalization
- **Optimizer**: Adam with adaptive learning rate
- **Loss Function**: Binary Cross-Entropy
- **Callbacks**: Early Stopping, Learning Rate Reduction

## SHAP Explainability

SHAP (SHapley Additive exPlanations) provides interpretable explanations for model predictions:

### Feature Importance
- **Mean |SHAP Value|**: Average absolute contribution of each feature
- **Summary Plot**: Distribution of SHAP values showing direction of impact

### Individual Explanations
- **Waterfall Plot**: Shows how each feature pushes the prediction higher or lower
- **Contribution Analysis**: Detailed breakdown of feature contributions

## Financial Features

The model analyzes 217 financial indicators, including:

### Raw Financial Items
- Balance Sheet: Cash, Receivables, Inventory, Fixed Assets, Liabilities, Equity
- Cash Flow: Operating, Investing, Financing cash flows
- Profit Statement: Revenue, Expenses, Operating Profit, Net Profit

### Calculated Ratios
- **Profitability**: ROA, ROE, Gross Margin, Operating Margin, Net Margin
- **Liquidity**: Current Ratio, Quick Ratio, Cash Ratio
- **Leverage**: Debt to Equity, Debt Ratio, Equity Ratio
- **Efficiency**: Asset Turnover, Receivables Turnover, Inventory Turnover
- **Cash Flow**: Operating Cash Flow to Debt, Cash Flow to Net Profit

### Risk Indicators
- **Altman Z-Score**: Bankruptcy prediction (Z < 1.8 = distress)
- **Beneish M-Score**: Earnings manipulation detection (M > -2.22 = potential manipulator)
- **Accruals Ratio**: Earnings quality indicator

## Fraud Labeling

Since the original dataset doesn't have explicit fraud labels, fraud is identified based on:

1. **Altman Z-Score** (Z < 1.8): Financial distress
2. **Beneish M-Score** (M > -2.22): Earnings manipulation
3. **High Accruals**: Potential earnings manipulation
4. **Cash Flow Anomalies**: Negative OCF with positive profit
5. **Leverage Extremes**: Debt ratio > 80%
6. **Negative Equity**: Technical insolvency
7. **Receivables Anomalies**: High receivables relative to revenue

## Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve

## Project Structure

```
financial_fraud_detection/
├── app.py                    # Main Streamlit application
├── data_processor.py         # Data loading and processing
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
└── financialreport.csv       # Processed dataset
```

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError"**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

2. **Memory issues**
   - Reduce SHAP sample size
   - Use smaller batch size

3. **Slow training**
   - Reduce epochs
   - Use GPU if available (install tensorflow-gpu)

## Future Improvements

- [ ] Add support for text-based features from annual reports
- [ ] Implement LSTM for time-series analysis
- [ ] Add more advanced ensemble methods
- [ ] Real-time data integration

## License

This project is provided for research purposes.

## Acknowledgments

- Dataset by [Ziya](https://www.kaggle.com/ziya07) on Kaggle
- SHAP library by Scott Lundberg
- TensorFlow/Keras team
- Streamlit team

---

<div align="center">
Made with ❤️ by Nada Marey
</div>
