"""
Financial Data Processor for Kaggle Dataset
============================================

This script processes the Kaggle Financial Reports Fraud Detection dataset,
creating a feature-rich dataset suitable for deep learning fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Chinese to English mapping for key financial items
FINANCIAL_ITEM_MAPPING = {
    # Balance Sheet Items
    'BalanceSheet_3': 'Cash',
    'BalanceSheet_4': 'TradingFinancialAssets',
    'BalanceSheet_5': 'DerivativeFinancialAssets',
    'BalanceSheet_6': 'NotesAndAccountsReceivable',
    'BalanceSheet_7': 'NotesReceivable',
    'BalanceSheet_8': 'AccountsReceivable',
    'BalanceSheet_9': 'AccountsReceivableFinancing',
    'BalanceSheet_10': 'Prepayments',
    'BalanceSheet_11': 'OtherReceivables',
    'BalanceSheet_12': 'InterestReceivable',
    'BalanceSheet_13': 'DividendReceivable',
    'BalanceSheet_14': 'OtherReceivables2',
    'BalanceSheet_15': 'BuybackResaleFinancialAssets',
    'BalanceSheet_16': 'Inventory',
    'BalanceSheet_17': 'HeldForSaleAssets',
    'BalanceSheet_18': 'NonCurrentAssetsDueWithinYear',
    'BalanceSheet_19': 'DeferredExpenses',
    'BalanceSheet_20': 'ProcessingCurrentAssetsLoss',
    'BalanceSheet_21': 'OtherCurrentAssets',
    'BalanceSheet_22': 'TotalCurrentAssets',
    'BalanceSheet_24': 'LoansAndAdvances',
    'BalanceSheet_25': 'AvailableForSaleFinancialAssets',
    'BalanceSheet_26': 'HeldToMaturityInvestments',
    'BalanceSheet_27': 'LongTermReceivables',
    'BalanceSheet_28': 'LongTermEquityInvestments',
    'BalanceSheet_29': 'InvestmentProperty',
    'BalanceSheet_30': 'ConstructionInProgress',
    'BalanceSheet_31': 'ConstructionInProgress2',
    'BalanceSheet_32': 'ConstructionMaterials',
    'BalanceSheet_33': 'FixedAssetsAndClearing',
    'BalanceSheet_34': 'FixedAssetsNet',
    'BalanceSheet_35': 'FixedAssetsClearing',
    'BalanceSheet_36': 'ProductiveBiologicalAssets',
    'BalanceSheet_37': 'PublicWelfareBiologicalAssets',
    'BalanceSheet_38': 'OilAndGasAssets',
    'BalanceSheet_39': 'RightOfUseAssets',
    'BalanceSheet_40': 'IntangibleAssets',
    'BalanceSheet_41': 'DevelopmentExpenditure',
    'BalanceSheet_42': 'Goodwill',
    'BalanceSheet_43': 'LongTermDeferredExpenses',
    'BalanceSheet_44': 'DeferredTaxAssets',
    'BalanceSheet_45': 'OtherNonCurrentAssets',
    'BalanceSheet_46': 'TotalNonCurrentAssets',
    'BalanceSheet_47': 'TotalAssets',
    'BalanceSheet_49': 'ShortTermBorrowings',
    'BalanceSheet_50': 'TradingFinancialLiabilities',
    'BalanceSheet_51': 'NotesAndAccountsPayable',
    'BalanceSheet_52': 'NotesPayable',
    'BalanceSheet_53': 'AccountsPayable',
    'BalanceSheet_54': 'AdvanceReceipts',
    'BalanceSheet_55': 'HandlingFeesAndCommissions',
    'BalanceSheet_56': 'EmployeeCompensationPayable',
    'BalanceSheet_57': 'TaxesPayable',
    'BalanceSheet_58': 'OtherPayables',
    'BalanceSheet_59': 'InterestPayable',
    'BalanceSheet_60': 'DividendPayable',
    'BalanceSheet_61': 'OtherPayables2',
    'BalanceSheet_62': 'AccruedExpenses',
    'BalanceSheet_63': 'DeferredRevenueWithinYear',
    'BalanceSheet_64': 'NonCurrentLiabilitiesDueWithinYear',
    'BalanceSheet_65': 'OtherCurrentLiabilities',
    'BalanceSheet_66': 'TotalCurrentLiabilities',
    'BalanceSheet_68': 'LongTermBorrowings',
    'BalanceSheet_69': 'BondsPayable',
    'BalanceSheet_70': 'LeaseLiabilities',
    'BalanceSheet_71': 'LongTermEmployeeCompensation',
    'BalanceSheet_72': 'LongTermPayables',
    'BalanceSheet_73': 'LongTermPayables2',
    'BalanceSheet_74': 'SpecificPayables',
    'BalanceSheet_75': 'EstimatedNonCurrentLiabilities',
    'BalanceSheet_76': 'DeferredTaxLiabilities',
    'BalanceSheet_77': 'LongTermDeferredRevenue',
    'BalanceSheet_78': 'OtherNonCurrentLiabilities',
    'BalanceSheet_79': 'TotalNonCurrentLiabilities',
    'BalanceSheet_80': 'TotalLiabilities',
    'BalanceSheet_82': 'PaidInCapital',
    'BalanceSheet_83': 'CapitalReserve',
    'BalanceSheet_84': 'LessTreasuryStock',
    'BalanceSheet_85': 'OtherComprehensiveIncome',
    'BalanceSheet_86': 'SpecialReserve',
    'BalanceSheet_87': 'SurplusReserve',
    'BalanceSheet_88': 'GeneralRiskReserve',
    'BalanceSheet_89': 'UndistributedProfit',
    'BalanceSheet_90': 'ParentCompanyEquity',
    'BalanceSheet_91': 'MinorityEquity',
    'BalanceSheet_92': 'TotalEquity',
    'BalanceSheet_93': 'TotalLiabilitiesAndEquity',
    
    # Cash Flow Items
    'CashFlow_3': 'CashFromSales',
    'CashFlow_4': 'TaxRefunds',
    'CashFlow_5': 'OtherOperatingCashReceived',
    'CashFlow_6': 'TotalOperatingCashInflow',
    'CashFlow_7': 'CashPaidForGoods',
    'CashFlow_8': 'CashPaidToEmployees',
    'CashFlow_9': 'TaxesPaid',
    'CashFlow_10': 'OtherOperatingCashPaid',
    'CashFlow_11': 'TotalOperatingCashOutflow',
    'CashFlow_12': 'NetOperatingCashFlow',
    'CashFlow_13': 'CashFromInvestmentWithdrawal',
    'CashFlow_14': 'CashFromInvestmentIncome',
    'CashFlow_15': 'CashFromAssetDisposal',
    'CashFlow_16': 'CashFromSubsidiaryDisposal',
    'CashFlow_17': 'OtherInvestingCashReceived',
    'CashFlow_18': 'TotalInvestingCashInflow',
    'CashFlow_19': 'CashPaidForAssets',
    'CashFlow_20': 'CashPaidForInvestment',
    'CashFlow_21': 'CashPaidForSubsidiaries',
    'CashFlow_22': 'OtherInvestingCashPaid',
    'CashFlow_23': 'TotalInvestingCashOutflow',
    'CashFlow_24': 'NetInvestingCashFlow',
    'CashFlow_25': 'CashFromInvestmentAbsorbed',
    'CashFlow_26': 'CashFromMinorityInvestment',
    'CashFlow_27': 'CashFromBorrowings',
    'CashFlow_28': 'CashFromBondIssuance',
    'CashFlow_29': 'OtherFinancingCashReceived',
    'CashFlow_30': 'TotalFinancingCashInflow',
    'CashFlow_31': 'CashPaidForDebt',
    'CashFlow_32': 'CashPaidForDividends',
    'CashFlow_33': 'CashPaidToMinority',
    'CashFlow_34': 'OtherFinancingCashPaid',
    'CashFlow_35': 'TotalFinancingCashOutflow',
    'CashFlow_36': 'NetFinancingCashFlow',
    
    # Profit Statement Items
    'ProfitStatement_3': 'TotalOperatingRevenue',
    'ProfitStatement_4': 'OperatingRevenue',
    'ProfitStatement_5': 'TotalOperatingCost',
    'ProfitStatement_6': 'OperatingCost',
    'ProfitStatement_7': 'TaxesAndSurcharges',
    'ProfitStatement_8': 'SalesExpenses',
    'ProfitStatement_9': 'AdminExpenses',
    'ProfitStatement_10': 'RDEnvironmentExpenses',
    'ProfitStatement_11': 'AssetImpairmentLoss',
    'ProfitStatement_12': 'FairValueChangeIncome',
    'ProfitStatement_13': 'InvestmentIncome',
    'ProfitStatement_14': 'InvestmentIncomeFromJoint',
    'ProfitStatement_15': 'OperatingProfit',
    'ProfitStatement_16': 'NonOperatingIncome',
    'ProfitStatement_17': 'NonOperatingExpenses',
    'ProfitStatement_18': 'TotalProfit',
    'ProfitStatement_19': 'IncomeTaxExpense',
    'ProfitStatement_20': 'NetProfit',
    'ProfitStatement_21': 'NetProfitToParent',
    'ProfitStatement_22': 'MinorityInterestLoss',
    'ProfitStatement_23': 'BasicEarningsPerShare',
    'ProfitStatement_24': 'DilutedEarningsPerShare',
    'ProfitStatement_25': 'OtherComprehensiveIncome',
    'ProfitStatement_26': 'TotalComprehensiveIncome',
    'ProfitStatement_27': 'ComprehensiveIncomeToParent',
    'ProfitStatement_28': 'ComprehensiveIncomeToMinority',
}


def load_and_process_data(data_path=None):
    """
    Load and process the financial dataset from local CSV file.
    
    Args:
        data_path: Path to the CSV file. If None, uses local financialreport.csv.
        
    Returns:
        Processed DataFrame with features and fraud labels
    """
    import os
    
    if data_path is None:
        # Use local financialreport.csv in the same directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'financialreport.csv')
    
    # Load raw data
    df = pd.read_csv(data_path, encoding='latin-1')
    
    # Create item key for pivoting
    df['item_key'] = df['report'] + '_' + df['sortid'].astype(str)
    
    # Pivot to wide format
    pivot_df = df.pivot_table(
        index=['company', 'period'],
        columns='item_key',
        values='amount',
        aggfunc='first'
    ).reset_index()
    
    # Rename columns
    pivot_df = pivot_df.rename(columns=FINANCIAL_ITEM_MAPPING)
    
    return pivot_df


def calculate_financial_ratios(df):
    """
    Calculate key financial ratios for fraud detection.
    
    Args:
        df: DataFrame with raw financial items
        
    Returns:
        DataFrame with additional financial ratio columns
    """
    df = df.copy()
    
    # Avoid division by zero
    eps = 1e-10
    
    # === Profitability Ratios ===
    
    # Gross Profit Margin
    if 'TotalOperatingRevenue' in df.columns and 'OperatingCost' in df.columns:
        df['GrossProfitMargin'] = (df['TotalOperatingRevenue'] - df['OperatingCost']) / (df['TotalOperatingRevenue'] + eps)
    
    # Operating Profit Margin
    if 'OperatingProfit' in df.columns and 'TotalOperatingRevenue' in df.columns:
        df['OperatingProfitMargin'] = df['OperatingProfit'] / (df['TotalOperatingRevenue'] + eps)
    
    # Net Profit Margin
    if 'NetProfit' in df.columns and 'TotalOperatingRevenue' in df.columns:
        df['NetProfitMargin'] = df['NetProfit'] / (df['TotalOperatingRevenue'] + eps)
    
    # ROA (Return on Assets)
    if 'NetProfit' in df.columns and 'TotalAssets' in df.columns:
        df['ROA'] = df['NetProfit'] / (df['TotalAssets'] + eps)
    
    # ROE (Return on Equity)
    if 'NetProfit' in df.columns and 'TotalEquity' in df.columns:
        df['ROE'] = df['NetProfit'] / (df['TotalEquity'] + eps)
    
    # === Liquidity Ratios ===
    
    # Current Ratio
    if 'TotalCurrentAssets' in df.columns and 'TotalCurrentLiabilities' in df.columns:
        df['CurrentRatio'] = df['TotalCurrentAssets'] / (df['TotalCurrentLiabilities'] + eps)
    
    # Quick Ratio (exclude inventory)
    if 'TotalCurrentAssets' in df.columns and 'Inventory' in df.columns and 'TotalCurrentLiabilities' in df.columns:
        df['QuickRatio'] = (df['TotalCurrentAssets'] - df['Inventory'].fillna(0)) / (df['TotalCurrentLiabilities'] + eps)
    
    # Cash Ratio
    if 'Cash' in df.columns and 'TotalCurrentLiabilities' in df.columns:
        df['CashRatio'] = df['Cash'] / (df['TotalCurrentLiabilities'] + eps)
    
    # === Leverage Ratios ===
    
    # Debt to Equity
    if 'TotalLiabilities' in df.columns and 'TotalEquity' in df.columns:
        df['DebtToEquity'] = df['TotalLiabilities'] / (df['TotalEquity'] + eps)
    
    # Debt Ratio
    if 'TotalLiabilities' in df.columns and 'TotalAssets' in df.columns:
        df['DebtRatio'] = df['TotalLiabilities'] / (df['TotalAssets'] + eps)
    
    # Equity Ratio
    if 'TotalEquity' in df.columns and 'TotalAssets' in df.columns:
        df['EquityRatio'] = df['TotalEquity'] / (df['TotalAssets'] + eps)
    
    # === Efficiency Ratios ===
    
    # Asset Turnover
    if 'TotalOperatingRevenue' in df.columns and 'TotalAssets' in df.columns:
        df['AssetTurnover'] = df['TotalOperatingRevenue'] / (df['TotalAssets'] + eps)
    
    # Receivables Turnover
    if 'TotalOperatingRevenue' in df.columns and 'AccountsReceivable' in df.columns:
        df['ReceivablesTurnover'] = df['TotalOperatingRevenue'] / (df['AccountsReceivable'] + eps)
    
    # Inventory Turnover
    if 'OperatingCost' in df.columns and 'Inventory' in df.columns:
        df['InventoryTurnover'] = df['OperatingCost'] / (df['Inventory'] + eps)
    
    # === Cash Flow Ratios ===
    
    # Operating Cash Flow to Debt
    if 'NetOperatingCashFlow' in df.columns and 'TotalLiabilities' in df.columns:
        df['OperatingCashFlowToDebt'] = df['NetOperatingCashFlow'] / (df['TotalLiabilities'] + eps)
    
    # Operating Cash Flow to Net Profit
    if 'NetOperatingCashFlow' in df.columns and 'NetProfit' in df.columns:
        df['CashFlowToNetProfit'] = df['NetOperatingCashFlow'] / (df['NetProfit'].abs() + eps)
    
    # Free Cash Flow
    if 'NetOperatingCashFlow' in df.columns and 'NetInvestingCashFlow' in df.columns:
        df['FreeCashFlow'] = df['NetOperatingCashFlow'] + df['NetInvestingCashFlow']
    
    # === Earnings Quality Indicators ===
    
    # Accruals Ratio (approximation)
    if 'NetProfit' in df.columns and 'NetOperatingCashFlow' in df.columns and 'TotalAssets' in df.columns:
        df['AccrualsRatio'] = (df['NetProfit'] - df['NetOperatingCashFlow']) / (df['TotalAssets'] + eps)
    
    # === Risk Scores ===
    
    # Altman Z-Score Components (approximation for Chinese companies)
    # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    # X1 = Working Capital / Total Assets
    # X2 = Retained Earnings / Total Assets  
    # X3 = EBIT / Total Assets
    # X4 = Market Equity / Total Liabilities (use Book Equity as proxy)
    # X5 = Sales / Total Assets
    
    if all(col in df.columns for col in ['TotalCurrentAssets', 'TotalCurrentLiabilities', 'TotalAssets']):
        df['WorkingCapital'] = df['TotalCurrentAssets'] - df['TotalCurrentLiabilities']
        df['WorkingCapitalToAssets'] = df['WorkingCapital'] / (df['TotalAssets'] + eps)
    
    if 'UndistributedProfit' in df.columns and 'TotalAssets' in df.columns:
        df['RetainedEarningsToAssets'] = df['UndistributedProfit'] / (df['TotalAssets'] + eps)
    
    if 'OperatingProfit' in df.columns and 'TotalAssets' in df.columns:
        df['EBITToAssets'] = df['OperatingProfit'] / (df['TotalAssets'] + eps)
    
    if 'TotalEquity' in df.columns and 'TotalLiabilities' in df.columns:
        df['EquityToLiabilities'] = df['TotalEquity'] / (df['TotalLiabilities'] + eps)
    
    if 'TotalOperatingRevenue' in df.columns and 'TotalAssets' in df.columns:
        df['SalesToAssets'] = df['TotalOperatingRevenue'] / (df['TotalAssets'] + eps)
    
    # Calculate Altman Z-Score (approximation)
    z_score_components = []
    if 'WorkingCapitalToAssets' in df.columns:
        z_score_components.append(1.2 * df['WorkingCapitalToAssets'])
    if 'RetainedEarningsToAssets' in df.columns:
        z_score_components.append(1.4 * df['RetainedEarningsToAssets'])
    if 'EBITToAssets' in df.columns:
        z_score_components.append(3.3 * df['EBITToAssets'])
    if 'EquityToLiabilities' in df.columns:
        z_score_components.append(0.6 * df['EquityToLiabilities'])
    if 'SalesToAssets' in df.columns:
        z_score_components.append(1.0 * df['SalesToAssets'])
    
    if z_score_components:
        df['AltmanZScore'] = sum(z_score_components)
    
    # Beneish M-Score Components (approximation)
    # M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    
    # Days Sales in Receivables Index (use receivables ratio as proxy)
    if 'AccountsReceivable' in df.columns and 'TotalOperatingRevenue' in df.columns:
        df['DaysSalesReceivables'] = df['AccountsReceivable'] / (df['TotalOperatingRevenue'] / 365 + eps)
    
    # Asset Quality Index (approximation)
    if all(col in df.columns for col in ['TotalAssets', 'TotalCurrentAssets', 'TotalNonCurrentAssets']):
        # Soft assets as proxy
        df['AssetQualityIndex'] = 1 - (df['TotalCurrentAssets'] + df.get('TotalNonCurrentAssets', 0)) / (df['TotalAssets'] + eps)
    
    # Leverage Index
    if 'DebtRatio' in df.columns:
        df['LeverageIndex'] = df['DebtRatio']
    
    # Calculate simplified Beneish M-Score
    m_score = -4.84
    if 'DaysSalesReceivables' in df.columns:
        m_score += 0.92 * (df['DaysSalesReceivables'] / df['DaysSalesReceivables'].median())
    if 'GrossProfitMargin' in df.columns:
        m_score += 0.528 * (1 - df['GrossProfitMargin'])
    if 'AssetQualityIndex' in df.columns:
        m_score += 0.404 * df['AssetQualityIndex']
    if 'SalesToAssets' in df.columns:
        m_score += 0.892 * df['SalesToAssets']
    if 'AccrualsRatio' in df.columns:
        m_score += 4.679 * df['AccrualsRatio'].abs()
    if 'LeverageIndex' in df.columns:
        m_score -= 0.327 * df['LeverageIndex']
    
    df['BeneishMScore'] = m_score
    
    return df


def create_fraud_labels(df):
    """
    Create fraud labels based on financial anomalies and risk indicators.
    
    Uses multiple criteria:
    1. Altman Z-Score (bankruptcy risk)
    2. Beneish M-Score (earnings manipulation)
    3. Abnormal financial ratios
    4. Cash flow anomalies
    
    Args:
        df: DataFrame with calculated financial ratios
        
    Returns:
        DataFrame with fraud label column
    """
    df = df.copy()
    
    # Initialize fraud probability
    fraud_score = np.zeros(len(df))
    
    # 1. Altman Z-Score contribution
    # Z < 1.8: Distress Zone (high risk)
    # 1.8 < Z < 2.99: Grey Zone
    # Z > 2.99: Safe Zone
    if 'AltmanZScore' in df.columns:
        fraud_score += np.where(df['AltmanZScore'] < 1.8, 0.3, 0)
        fraud_score += np.where(df['AltmanZScore'] < 1.0, 0.2, 0)
    
    # 2. Beneish M-Score contribution
    # M > -2.22: Potential manipulator
    if 'BeneishMScore' in df.columns:
        fraud_score += np.where(df['BeneishMScore'] > -2.22, 0.25, 0)
        fraud_score += np.where(df['BeneishMScore'] > 0, 0.15, 0)
    
    # 3. Accruals anomaly
    # High positive accruals can indicate earnings manipulation
    if 'AccrualsRatio' in df.columns:
        fraud_score += np.where(df['AccrualsRatio'] > 0.1, 0.15, 0)
        fraud_score += np.where(df['AccrualsRatio'] > 0.2, 0.1, 0)
    
    # 4. Profit quality
    # Negative operating cash flow with positive net profit is suspicious
    if 'NetOperatingCashFlow' in df.columns and 'NetProfit' in df.columns:
        fraud_score += np.where(
            (df['NetOperatingCashFlow'] < 0) & (df['NetProfit'] > 0), 0.2, 0
        )
    
    # 5. Revenue growth vs cash flow
    # High revenue growth with negative cash flow is suspicious
    if 'TotalOperatingRevenue' in df.columns and 'NetOperatingCashFlow' in df.columns:
        # Calculate year-over-year revenue growth (approximation)
        fraud_score += np.where(
            (df['TotalOperatingRevenue'] > df['TotalOperatingRevenue'].median()) & 
            (df['NetOperatingCashFlow'] < 0), 0.1, 0
        )
    
    # 6. Leverage extremes
    if 'DebtRatio' in df.columns:
        fraud_score += np.where(df['DebtRatio'] > 0.8, 0.15, 0)
        fraud_score += np.where(df['DebtRatio'] > 0.9, 0.1, 0)
    
    # 7. Negative equity
    if 'TotalEquity' in df.columns:
        fraud_score += np.where(df['TotalEquity'] < 0, 0.3, 0)
    
    # 8. Extreme profitability margins
    if 'NetProfitMargin' in df.columns:
        fraud_score += np.where(df['NetProfitMargin'] > 0.5, 0.1, 0)  # Suspiciously high
        fraud_score += np.where(df['NetProfitMargin'] < -0.3, 0.15, 0)  # Severe losses
    
    # 9. Receivables anomalies
    # High receivables relative to revenue could indicate fake sales
    if 'AccountsReceivable' in df.columns and 'TotalOperatingRevenue' in df.columns:
        receivables_ratio = df['AccountsReceivable'] / (df['TotalOperatingRevenue'].abs() + 1e-10)
        fraud_score += np.where(receivables_ratio > 0.5, 0.15, 0)
        fraud_score += np.where(receivables_ratio > 1.0, 0.1, 0)
    
    # Normalize fraud score to probability
    fraud_score = np.clip(fraud_score / 1.5, 0, 1)
    
    # Create binary label with threshold
    # Higher threshold = more conservative (fewer fraud cases)
    df['FraudProbability'] = fraud_score
    df['Fraud'] = (fraud_score > 0.5).astype(int)
    
    # Add some randomness to avoid perfect separation
    np.random.seed(42)
    random_noise = np.random.uniform(-0.1, 0.1, len(df))
    df['FraudProbability'] = np.clip(fraud_score + random_noise, 0, 1)
    
    return df


def prepare_model_features(df):
    """
    Prepare features for model training.
    
    Args:
        df: Processed DataFrame with all features
        
    Returns:
        Tuple of (feature DataFrame, feature names list)
    """
    # Define feature columns (exclude identifiers and labels)
    exclude_cols = ['company', 'period', 'Fraud', 'FraudProbability']
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create feature DataFrame
    X = df[feature_cols].copy()
    
    # Handle inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    return X, feature_cols


def process_full_dataset(data_path=None, output_path=None):
    """
    Full pipeline to process the Kaggle dataset.
    
    Args:
        data_path: Path to raw data
        output_path: Path to save processed data
        
    Returns:
        Processed DataFrame
    """
    print("Loading and processing data...")
    df = load_and_process_data(data_path)
    print(f"  Pivoted shape: {df.shape}")
    
    print("Calculating financial ratios...")
    df = calculate_financial_ratios(df)
    print(f"  After ratios: {df.shape}")
    
    print("Creating fraud labels...")
    df = create_fraud_labels(df)
    
    fraud_count = df['Fraud'].sum()
    fraud_rate = fraud_count / len(df) * 100
    print(f"  Fraud cases: {fraud_count} ({fraud_rate:.1f}%)")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    import os
    
    # Process the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'processed_financial_data.csv')
    
    df = process_full_dataset(output_path=output_path)
    
    print("\nDataset summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Total features: {len(df.columns) - 4}")  # Exclude company, period, Fraud, FraudProbability
    print(f"  Companies: {df['company'].nunique()}")
    print(f"  Fraud rate: {df['Fraud'].mean()*100:.1f}%")
    
    print("\nFeature columns:")
    feature_cols = [col for col in df.columns if col not in ['company', 'period', 'Fraud', 'FraudProbability']]
    print(f"  {len(feature_cols)} features")
    
    print("\nSample data:")
    print(df.head())
