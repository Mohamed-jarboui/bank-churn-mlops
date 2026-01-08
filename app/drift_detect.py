"""
Data Drift Detection Module
Compares production data against reference data to detect distribution shifts.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any


def detect_drift(
    reference_file: str,
    production_file: str,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Detect data drift between reference and production datasets.
    
    Args:
        reference_file: Path to reference dataset CSV
        production_file: Path to production dataset CSV
        threshold: P-value threshold for drift detection (default: 0.05)
    
    Returns:
        Dictionary with drift detection results for each feature
    """
    
    # Load datasets
    try:
        ref_data = pd.read_csv(reference_file)
        prod_data = pd.read_csv(production_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e}")
    
    results = {}
    
    # Features to check for drift
    features = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
        'EstimatedSalary', 'Geography_Germany', 'Geography_Spain'
    ]
    
    for feature in features:
        if feature not in ref_data.columns or feature not in prod_data.columns:
            continue
        
        ref_values = ref_data[feature].dropna()
        prod_values = prod_data[feature].dropna()
        
        # Determine if feature is continuous or categorical
        is_continuous = ref_data[feature].dtype in ['float64', 'int64'] and len(ref_values.unique()) > 10
        
        if is_continuous:
            # Use Kolmogorov-Smirnov test for continuous features
            statistic, p_value = stats.ks_2samp(ref_values, prod_values)
            test_type = "ks_test"
        else:
            # Use Chi-square test for categorical features
            ref_freq = ref_values.value_counts(normalize=True)
            prod_freq = prod_values.value_counts(normalize=True)
            
            # Align indices
            all_categories = set(ref_freq.index) | set(prod_freq.index)
            ref_freq = ref_freq.reindex(all_categories, fill_value=0)
            prod_freq = prod_freq.reindex(all_categories, fill_value=0)
            
            # Perform chi-square test
            observed = prod_freq * len(prod_values)
            expected = ref_freq * len(prod_values)
            
            # Add small constant to avoid division by zero
            expected = expected + 1e-10
            
            chi2_stat = np.sum((observed - expected) ** 2 / expected)
            df = len(all_categories) - 1
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)
            statistic = chi2_stat
            test_type = "chi2_test"
        
        # Drift detected if p-value < threshold
        drift_detected = p_value < threshold
        
        results[feature] = {
            "drift_detected": drift_detected,
            "p_value": p_value,
            "statistic": statistic,
            "type": test_type,
            "threshold": threshold
        }
    
    return results
