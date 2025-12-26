import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
TARGET = "loan_paid_back"   # change as needed

NUMS = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

#Digit extraction function
def add_digit_features(df, cols, positions=[-3, -2, -1, 0, 1]):
    for c in cols:
        for k in positions:
            df[f"{c}_d{k}"] = ((df[c] * (10**k)) % 10).astype('int')
    return df

# Rounding function
def add_rounding(df, cols, round_levels=[-2, -1, 0, 1]):
    for c in cols:
        for r in round_levels:
            df[f"{c}_r{r}"] = df[c].round(r)
    return df
# Binning function
def add_binning(train, test, cols, q=5):
    for c in cols:
        try:
            train[f"{c}_bin"] = pd.qcut(train[c], q=q, labels=False, duplicates='drop')
            bins = train[f"{c}_bin"].cat.categories
            test[f"{c}_bin"] = pd.cut(test[c], bins=bins, labels=False).astype(float)
        except:
            train[f"{c}_bin"] = 0
            test[f"{c}_bin"] = 0
    return train, test

#Target encoding function K-fold
def kfold_target_encoding(train, test, col, target, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = train[target].mean()
    
    train[f"TE_{col}"] = np.nan
    test[f"TE_{col}"] = 0
    
    # Out-of-fold TE for train
    for tr_idx, val_idx in kf.split(train, train[target]):
        means = train.iloc[tr_idx].groupby(col)[target].mean()
        train.loc[val_idx, f"TE_{col}"] = train.loc[val_idx, col].map(means)
    
    train[f"TE_{col}"] = train[f"TE_{col}"].fillna(global_mean)
    
    # TE for test
    full_means = train.groupby(col)[target].mean()
    test[f"TE_{col}"] = test[col].map(full_means).fillna(global_mean)
    
    return train, test

#count encoding function
def count_encoding(train, test, col):
    counts = train[col].value_counts()
    train[f"CE_{col}"] = train[col].map(counts).fillna(0)
    test[f"CE_{col}"] = test[col].map(counts).fillna(0)
    return train, test
# Interaction features function Pairwise
def add_interactions(df, cols):
    from itertools import combinations
    for c1, c2 in combinations(cols, 2):
        df[f"{c1}__{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df

def full_feature_engineering(train, test):
    # 1. Basic features
    train, test = add_binning(train, test, NUMS)
    train = add_digit_features(train, NUMS)
    test = add_digit_features(test, NUMS)
    train = add_rounding(train, NUMS)
    test = add_rounding(test, NUMS)
    
    # 2. Interactions (categoricals only)
    train = add_interactions(train, CATS)
    test = add_interactions(test, CATS)
    
    # 3. Apply TE + CE
    for c in CATS + NUMS:
        train, test = kfold_target_encoding(train, test, c, TARGET)
        train, test = count_encoding(train, test, c)
    
    # 4. Interactions TE/CE (optional, strong but heavier)
    # for c in train.filter(regex="__").columns:
    #     train, test = kfold_target_encoding(train, test, c, TARGET)

    # 5. Downcast to save memory
    train = train.copy()
    test = test.copy()
    for col in train.columns:
        if train[col].dtype == "float64":
            train[col] = train[col].astype("float32")
            test[col] = test[col].astype("float32")
        if train[col].dtype == "int64":
            train[col] = train[col].astype("int32")
            test[col] = test[col].astype("int32")
    
    return train, test
