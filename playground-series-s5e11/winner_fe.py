# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================
import pandas as pd
import numpy as np
import warnings
import gc
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from itertools import combinations
from xgboost import XGBClassifier
from scipy.special import expit
from tqdm import tqdm

# Configure display options for debugging
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option("display.max_colwidth", None)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# ============================================================================
# FEATURE DEFINITIONS AND DATA LOADING
# ============================================================================
# Target variable for binary classification
TARGET = 'loan_paid_back'

# Numerical features: continuous/quantitative variables
NUMS = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

# Categorical features: discrete/qualitative variables
CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

# Load training and test datasets
train = pd.read_csv('train.csv', index_col='id')
test = pd.read_csv('test.csv', index_col='id')

# ============================================================================
# FEATURE ENGINEERING 1: BINNED FEATURES (QUANTILE-BASED BUCKETING)
# ============================================================================
# Create binned versions of numerical features using quantiles (5 bins)
# This converts continuous variables into discrete categorical bins
bin_features_train = pd.DataFrame(index=train.index)
bin_features_test = pd.DataFrame(index=test.index)

for c in NUMS:
    for q in [5]:
        try:
            # Use pd.qcut to create equal-frequency bins on training data
            train_bins, bins = pd.qcut(train[c], q=q, labels=False, retbins=True, duplicates="drop")
            bin_features_train[f"{c}_bin{q}"] = train_bins
            # Apply same bin edges to test data
            bin_features_test[f"{c}_bin{q}"] = pd.cut(test[c], bins=bins, labels=False, include_lowest=True)
        except Exception:
            # Fallback: if binning fails, create constant column (all 0s)
            bin_features_train[f"{c}_bin{q}"] = 0
            bin_features_test[f"{c}_bin{q}"] = 0

# Merge binned features with main datasets
train = pd.concat([train, bin_features_train], axis=1)
test = pd.concat([test, bin_features_test], axis=1)

# ============================================================================
# FEATURE ENGINEERING 2: DEFAULT RISK COMPOSITE SCORE
# ============================================================================
# Create a weighted composite risk score based on domain knowledge
# Weights: Debt-to-income (40%) + Credit Risk (35%) + Interest Rate (25%)
train['default_risk'] = (train['debt_to_income_ratio'] * 0.40 + (850 - train['credit_score']) / 850 * 0.35 + train['interest_rate'] / 100 * 0.25)
test['default_risk'] = (test['debt_to_income_ratio'] * 0.40 + (850 - test['credit_score']) / 850 * 0.35 + test['interest_rate'] / 100 * 0.25)

# Create a copy of credit_score for independent feature engineering
for c in ['credit_score']:
    n = f'{c}2'
    train[n] = train[c].copy()
    test[n] = test[c].copy()

# ============================================================================
# FEATURE ENGINEERING 3: DIGIT EXTRACTION FEATURES
# ============================================================================
# Extract individual digits at different decimal positions
# This captures patterns in specific digit positions (e.g., last digit, thousands digit)
DIGITS = []

# For income and loan amount: extract digits from -4 to 1 decimal places
for c in ['annual_income', 'loan_amount']:
    for k in range(-4, 2):
        n = f'{c}_d{k}'
        # Extract digit at position 10^k and handle NaN values
        train[n] = ((train[c] * 10**k) % 10).fillna(-1).astype("int8")
        test[n] = ((test[c] * 10**k) % 10).fillna(-1).astype("int8")
        DIGITS.append(n)

# For interest rate: extract digits from -1 to 2 decimal places
for c in ['interest_rate']:
    for k in range(-1, 3):
        n = f'{c}_d{k}'
        train[n] = ((train[c] * 10**k) % 10).fillna(-1).astype("int8")
        test[n] = ((test[c] * 10**k) % 10).fillna(-1).astype("int8")
        DIGITS.append(n)

# For debt ratio: extract digits from 1st to 3rd decimal places
for c in ['debt_to_income_ratio']:
    for k in range(1, 4):
        n = f'{c}_d{k}'
        train[n] = ((train[c] * 10**k) % 10).fillna(-1).astype("int8")
        test[n] = ((test[c] * 10**k) % 10).fillna(-1).astype("int8")
        DIGITS.append(n)

# Extract the subgrade digit from grade_subgrade (e.g., "A1" -> 1)
train['grade_subgrade_d1'] = train['grade_subgrade'].apply(lambda x: x[1]).astype('int8')
test['grade_subgrade_d1'] = test['grade_subgrade'].apply(lambda x: x[1]).astype('int8')

# ============================================================================
# FEATURE ENGINEERING 4: ROUNDING FEATURES
# ============================================================================
# Create features by rounding to different decimal places
# Captures magnitude/scale patterns
ROUND = []
RR = [-1, 0]  # Round to nearest 10 and nearest 1

for c in ['annual_income', 'loan_amount']:
    for r in RR:
        n = f"{c}_r{r}"
        train[n] = train[c].round(r)
        test[n] = test[c].round(r)
        ROUND.append(n)

# ============================================================================
# FEATURE ENGINEERING 5: LABEL ENCODING OF CATEGORICAL FEATURES
# ============================================================================
# Convert categorical features to numeric codes (0, 1, 2, ...)
# Ensures consistent encoding between train and test
for c in CATS + ['credit_score2']:
    combined = pd.concat([train[c], test[c]])
    combined, _ = combined.factorize()  # Get numeric codes
    train[c] = combined[:len(train)]
    test[c] = combined[len(train):len(train) + len(test)]

# ============================================================================
# FEATURE ENGINEERING 6: BIGRAM (2-WAY) COMBINATIONS
# ============================================================================
# Create interaction features by combining pairs of columns
# Captures relationships between features
TE_columns = []  # Columns to be target encoded
CE_columns = []  # Columns to be count encoded
PAIRS = []

columns = NUMS + CATS + [ROUND[0]]

# Generate all 2-way combinations
for r in [2]:
    for cols in tqdm(list(combinations(columns, r))):
        name = '-'.join(cols)

        # Concatenate column values as strings to create interaction feature
        train[name] = train[cols[0]].astype(str)
        for col in cols[1:]:
            train[name] = train[name] + '_' + train[col].astype(str)

        test[name] = test[cols[0]].astype(str)
        for col in cols[1:]:
            test[name] = test[name] + '_' + test[col].astype(str)

        # Label encode the combined feature
        combined = pd.concat([train[name], test[name]], ignore_index=True)
        combined, _ = combined.factorize()
        
        # Drop feature if cardinality is too high (more than 50% unique values)
        if pd.Series(combined).nunique() > len(combined) // 2:
            train = train.drop(name, axis=1)
            test = test.drop(name, axis=1)
            continue
        
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]
        TE_columns.append(name)
        CE_columns.append(name)
        PAIRS.append(name)

# ============================================================================
# FEATURE ENGINEERING 7: DIGIT + CATEGORICAL COMBINATIONS
# ============================================================================
# Combine digit features with specific categorical/numerical features
for c1 in DIGITS[:6]:
    for c2 in ['employment_status', 'debt_to_income_ratio']:
        name = f'{c1}-{c2}'
        train[name] = train[c1].astype(str) + '_' + train[c2].astype(str)
        test[name] = test[c1].astype(str) + '_' + test[c2].astype(str)
    
        combined = pd.concat([train[name], test[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]

        TE_columns.append(name)

# Combine pairs of digit features
for c1 in DIGITS[:6]:
    for c2 in [DIGITS[6], DIGITS[7]]:
        name = f'{c1}-{c2}'
        train[name] = train[c1].astype(str) + '_' + train[c2].astype(str)
        test[name] = test[c1].astype(str) + '_' + test[c2].astype(str)

        combined = pd.concat([train[name], test[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]

        TE_columns.append(name)

# ============================================================================
# FEATURE ENGINEERING 8: MULTI-WAY COMBINATION (3 FEATURES)
# ============================================================================
# Create a specific 3-way interaction: annual_income + gender + marital_status
for cols in tqdm([['annual_income', 'gender', 'marital_status']]):
    name = '-'.join(cols)

    train[name] = train[cols[0]].astype(str)
    for col in cols[1:]:
        train[name] = train[name] + '_' + train[col].astype(str)

    test[name] = test[cols[0]].astype(str)
    for col in cols[1:]:
        test[name] = test[name] + '_' + test[col].astype(str)

    combined = pd.concat([train[name], test[name]], ignore_index=True)
    combined, _ = combined.factorize()
    train[name] = combined[:len(train)]
    test[name] = combined[len(train):len(train) + len(test)]

    TE_columns.append(name)

# ============================================================================
# FEATURE ENGINEERING 9: K-FOLD TARGET ENCODING (PRIMARY TARGET)
# ============================================================================
# Target encoding: replace category with mean target value
# Uses K-Fold cross-validation to avoid overfitting to train data
TE_ORIG = []
CC = CATS + NUMS + DIGITS[:16]  # Features to target encode

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Processing {len(CC)} columns using K-Fold TE... ", end="")

for i, c in enumerate(CC):
    if i % 10 == 0:
        print(f"{i}, ", end="")
    
    # Create target encoding for predicting loan_paid_back
    # Out-of-fold: use fold means to avoid leakage
    train[f'TE_KFOLD_{c}'] = np.nan
    test[f'TE_KFOLD_{c}'] = 0
    
    # Global mean for handling unseen categories
    global_mean = train[TARGET].mean()
    
    # Out-of-fold encoding: each fold uses mean from other folds
    for train_idx, val_idx in kf.split(train, train[TARGET]):
        X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
        
        # Calculate category means using training fold only
        means = X_tr.groupby(c)[TARGET].mean()
        
        # Apply means to validation fold
        train.loc[val_idx, f'TE_KFOLD_{c}'] = train.loc[val_idx, c].map(means)
    
    # Fill NaN with global mean (unseen categories in validation)
    train[f'TE_KFOLD_{c}'] = train[f'TE_KFOLD_{c}'].fillna(global_mean)
    
    # Test encoding: use full training data means
    full_means = train.groupby(c)[TARGET].mean()
    test[f'TE_KFOLD_{c}'] = test[c].map(full_means).fillna(global_mean)
    
    # Count encoding: frequency of each category
    counts = train[c].value_counts()
    train[f'CE_{c}'] = train[c].map(counts).fillna(0)
    test[f'CE_{c}'] = test[c].map(counts).fillna(0)

# ============================================================================
# FEATURE ENGINEERING 10: TARGET ENCODING WITH EMPLOYMENT_STATUS AS TARGET
# ============================================================================
# Create features that predict employment_status instead of loan_paid_back
# Similar K-fold strategy but different target variable
CC = CATS + NUMS

print(f"Processing employment_status TE for {len(CATS + NUMS)} columns... ", end="")

for i, c in enumerate(CATS + NUMS):
    if i % 10 == 0:
        print(f"{i}, ", end="")
    
    global_mean = train['employment_status'].mean()
    
    train[f'TE_emp_{c}'] = np.nan
    test[f'TE_emp_{c}'] = 0
    
    # Out-of-fold target encoding using employment_status
    for tr_idx, val_idx in kf.split(train, train[TARGET]):  
        X_tr, X_val = train.iloc[tr_idx], train.iloc[val_idx]
        
        # Calculate mean employment_status for each category
        means = X_tr.groupby(c)['employment_status'].mean()
        
        train.loc[val_idx, f'TE_emp_{c}'] = train.loc[val_idx, c].map(means)
    
    # Fill missing values
    train[f'TE_emp_{c}'] = train[f'TE_emp_{c}'].fillna(global_mean)
    
    # Apply to test using full training data
    full_means = train.groupby(c)['employment_status'].mean()
    test[f'TE_emp_{c}'] = test[c].map(full_means).fillna(global_mean)

print()

# ============================================================================
# FEATURE ENGINEERING 11: MULTI-WAY DIGIT COMBINATIONS
# ============================================================================
# Create 2-way, 3-way, and 4-way combinations of digit features
# Captures complex interactions in numeric patterns
DIGIT_PAIRS = []

# 2, 3, 4-way combinations of first 6 digit features
for r in [2, 3, 4]:
    for cols in tqdm(list(combinations(DIGITS[:6], r))):
        name = '-'.join(cols)

        # Concatenate column values as strings to create interaction feature
        train[name] = train[cols[0]].astype(str)
        for col in cols[1:]:
            train[name] = train[name] + '_' + train[col].astype(str)

        test[name] = test[cols[0]].astype(str)
        for col in cols[1:]:
            test[name] = test[name] + '_' + test[col].astype(str)

        combined = pd.concat([train[name], test[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]
        DIGIT_PAIRS.append(name)
        TE_columns.append(name)
        CE_columns.append(name)

# 2, 3, 4-way combinations of digit features 6-11 (different numeric feature)
for r in [2, 3, 4]:
    for cols in tqdm(list(combinations(DIGITS[6:12], r))):
        name = '-'.join(cols)

        # Concatenate column values as strings to create interaction feature
        train[name] = train[cols[0]].astype(str)
        for col in cols[1:]:
            train[name] = train[name] + '_' + train[col].astype(str)

        test[name] = test[cols[0]].astype(str)
        for col in cols[1:]:
            test[name] = test[name] + '_' + test[col].astype(str)

        combined = pd.concat([train[name], test[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]
        DIGIT_PAIRS.append(name)
        TE_columns.append(name)
        CE_columns.append(name)

# 2, 3, 4-way combinations of digit features 12-15 (another numeric feature)
for r in [2, 3, 4]:
    for cols in tqdm(list(combinations(DIGITS[12:16], r))):
        name = '-'.join(cols)

        # Concatenate column values as strings to create interaction feature
        train[name] = train[cols[0]].astype(str)
        for col in cols[1:]:
            train[name] = train[name] + '_' + train[col].astype(str)

        test[name] = test[cols[0]].astype(str)
        for col in cols[1:]:
            test[name] = test[name] + '_' + test[col].astype(str)

        combined = pd.concat([train[name], test[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]
        DIGIT_PAIRS.append(name)
        TE_columns.append(name)
        CE_columns.append(name)

# 2, 3-way combinations of digit features 16-18 (interest rate digits)
for r in [2, 3]:
    for cols in tqdm(list(combinations(DIGITS[16:19], r))):
        name = '-'.join(cols)

        # Concatenate column values as strings to create interaction feature
        train[name] = train[cols[0]].astype(str)
        for col in cols[1:]:
            train[name] = train[name] + '_' + train[col].astype(str)

        test[name] = test[cols[0]].astype(str)
        for col in cols[1:]:
            test[name] = test[name] + '_' + test[col].astype(str)

        combined = pd.concat([train[name], test[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]
        DIGIT_PAIRS.append(name)
        TE_columns.append(name)
        CE_columns.append(name)

# ============================================================================
# FEATURE ENGINEERING 12: MEAN TARGET ENCODING BY SECONDARY FEATURES
# ============================================================================
# Create features that capture mean values of other features grouped by each feature
# Helps the model understand relationships between variables

# Mean employment_status grouped by each numeric/categorical feature
for c in NUMS + CATS:
    if c != 'employment_status':
        tmp = train.groupby(c)['employment_status'].mean()
        tmp.name = f'TE_mean_(employment_status)_{c}'
        train = train.merge(tmp, on=c, how='left')
        train[tmp.name] = train[tmp.name].fillna(train[tmp.name].mean())
        test = test.merge(tmp, on=c, how='left')
        test[tmp.name] = test[tmp.name].fillna(train[tmp.name].mean())

# Mean debt_to_income_ratio grouped by each numeric/categorical feature
for c in NUMS + CATS:
    if c != 'debt_to_income_ratio':
        tmp = train.groupby(c)['debt_to_income_ratio'].mean()
        tmp.name = f'TE_mean_(debt_to_income_ratio)_{c}'
        train = train.merge(tmp, on=c, how='left')
        train[tmp.name] = train[tmp.name].fillna(train[tmp.name].mean())
        test = test.merge(tmp, on=c, how='left')
        test[tmp.name] = test[tmp.name].fillna(train[tmp.name].mean())

# ============================================================================
# DATA TYPE OPTIMIZATION
# ============================================================================
# Convert float64/int64 to float32/int32 to reduce memory usage
for c in test.columns.tolist():
    if test[c].dtype == 'float64':
        train[c] = train[c].astype('float32')
        test[c] = test[c].astype('float32')
    if test[c].dtype == 'int64':
        train[c] = train[c].astype('int32')
        test[c] = test[c].astype('int32')

# ============================================================================
# FINAL FEATURE LIST
# ============================================================================
# Create list of all features excluding target variable
FEATURES = train.columns.tolist()
FEATURES.remove(TARGET)

def count_encode(train, valid, test, col):
    counts = train[col].value_counts()

    train[f'CE_{col}'] = train[col].map(counts)
    valid[f'CE_{col}'] = valid[col].map(counts).fillna(0)
    test[f'CE_{col}'] = test[col].map(counts).fillna(0)
    return (train, valid, test)

