# Cleaned diabetes model training pipeline
# - Removed notebook fluff/markdown
# - Kept full pipeline: FE, target encoding, XGB/LGBM/CatBoost folds
# - Comments are concise and focused on intent and non-obvious logic

import gc
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# Optional plotting (kept minimal)
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Configuration / I/O
# ----------------------
TARGET = 'diagnosed_diabetes'
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'
ORIG_CSV = 'diabetes_dataset.csv'  # external dataset used for orig-features

# ----------------------
# Utilities
# ----------------------

def reduce_mem_usage(df):
    """Downcast numeric dtypes to reduce memory footprint in-place."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object or str(col_type).startswith('category'):
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if str(col_type).startswith('int'):
            if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df


# ----------------------
# Target Encoder
# ----------------------
class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder supporting multiple aggregations and internal CV smoothing.
    - `aggs`: list of pandas-agg names (e.g., ['mean','count']).
    - Smoothing is applied only to the 'mean' aggregation; use 'auto' to
      compute an empirical Bayes-like m, otherwise pass numeric `m`.
    """
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        """Compute full-data mappings for use on unseen data (test/val).
        These mappings are conservative: unseen categories map to global stat."""
        temp = X.copy()
        temp['target'] = y
        for agg in self.aggs:
            self.global_stats_[agg] = y.agg(agg)
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg in self.aggs:
                self.mappings_[col][agg] = temp.groupby(col)['target'].agg(agg)
        return self

    def transform(self, X):
        """Map categories using fitted mappings; fill unseen with global stat."""
        X_out = X.copy()
        for col in self.cols_to_encode:
            for agg in self.aggs:
                name = f'TE_{col}_{agg}'
                mapping = self.mappings_[col][agg]
                X_out[name] = X_out[col].map(mapping)
                X_out[name].fillna(self.global_stats_[agg], inplace=True)
        if self.drop_original:
            X_out.drop(columns=self.cols_to_encode, inplace=True)
        return X_out

    def fit_transform(self, X, y):
        """
        Perform internal CV to produce out-of-fold target-encoded columns.
        This prevents leakage by computing group stats only on training folds.
        """
        self.fit(X, y)  # store global mappings for later `transform` calls
        encoded = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            tr_temp = X_tr.copy()
            tr_temp['target'] = y_tr

            for col in self.cols_to_encode:
                for agg in self.aggs:
                    name = f'TE_{col}_{agg}'
                    fold_global = y_tr.agg(agg)
                    mapping = tr_temp.groupby(col)['target'].agg(agg)

                    if agg == 'mean':
                        counts = tr_temp.groupby(col)['target'].count()
                        m = self.smooth
                        if self.smooth == 'auto':
                            # Empirical Bayes heuristic: ratio of within/between group variance
                            between = mapping.var()
                            within = tr_temp.groupby(col)['target'].var().mean()
                            m = (within / between) if between > 0 else 0
                        smoothed = (counts * mapping + m * fold_global) / (counts + m)
                        vals = X_val[col].map(smoothed)
                    else:
                        vals = X_val[col].map(mapping)

                    # fill unseen categories with fold-global stat (conservative)
                    encoded.loc[X_val.index, name] = vals.fillna(fold_global)

        # merge encoded features into copy of X
        X_out = X.copy()
        for c in encoded.columns:
            X_out[c] = encoded[c]
        if self.drop_original:
            X_out.drop(columns=self.cols_to_encode, inplace=True)
        return X_out


# ----------------------
# Feature engineering
# ----------------------

def build_features(train, test, orig):
    """Create features used by all models. Returns (X, y, FEATURES, CATS, NUMS, interaction_cols).
    Keeps operations deterministic and ensures train/test share categorical encodings where needed."""
    TARGET = 'diagnosed_diabetes'

    # Base features present in train (exclude target)
    BASE = [c for c in train.columns if c != TARGET]

    # Categorical and numeric separation (object types treated as categorical)
    CATS = train.select_dtypes('object').columns.to_list()
    NUMS = [c for c in BASE if c not in CATS]

    # Use `orig` (external dataset) to produce group mean/count features when possible
    ORIG_feats = []
    common_cols = [c for c in BASE if c in orig.columns]
    for col in common_cols:
        mean_map = orig.groupby(col)[TARGET].mean().reset_index(name=f'orig_mean_{col}')
        train = train.merge(mean_map, on=col, how='left')
        test = test.merge(mean_map, on=col, how='left')
        ORIG_feats.append(f'orig_mean_{col}')

        count_map = orig.groupby(col).size().reset_index(name=f'orig_count_{col}')
        train = train.merge(count_map, on=col, how='left')
        test = test.merge(count_map, on=col, how='left')
        ORIG_feats.append(f'orig_count_{col}')

    # Fill NAs: means -> global mean; counts -> 0
    for col in ORIG_feats:
        if 'mean' in col:
            train[col].fillna(orig[TARGET].mean(), inplace=True)
            test[col].fillna(orig[TARGET].mean(), inplace=True)
        else:
            train[col].fillna(0, inplace=True)
            test[col].fillna(0, inplace=True)

    FEATURES = BASE + ORIG_feats

    # Binning examples: concise, reproducible bins
    train['bmi_bin'] = pd.cut(train['bmi'], bins=[0, 25, 30, 100], labels=['normal', 'overweight', 'obese'])
    test['bmi_bin'] = pd.cut(test['bmi'], bins=[0, 25, 30, 100], labels=['normal', 'overweight', 'obese'])
    CATS.append('bmi_bin'); FEATURES.append('bmi_bin')

    train['age_bin'] = pd.cut(train['age'], bins=[0, 40, 50, 60, 120], labels=['<40', '40-50', '50-60', '60+'])
    test['age_bin'] = pd.cut(test['age'], bins=[0, 40, 50, 60, 120], labels=['<40', '40-50', '50-60', '60+'])
    CATS.append('age_bin'); FEATURES.append('age_bin')

    # Activity bins with clipping to remove extreme outliers
    train['activity_clipped'] = train['physical_activity_minutes_per_week'].clip(0, 300)
    test['activity_clipped'] = test['physical_activity_minutes_per_week'].clip(0, 300)
    train['activity_bin'] = pd.cut(train['activity_clipped'], bins=[0, 60, 150, 1000], labels=['low', 'moderate', 'high'])
    test['activity_bin'] = pd.cut(test['activity_clipped'], bins=[0, 60, 150, 1000], labels=['low', 'moderate', 'high'])
    train.drop(columns=['activity_clipped'], inplace=True)
    test.drop(columns=['activity_clipped'], inplace=True)
    CATS.append('activity_bin'); FEATURES.append('activity_bin')

    # Create low-cardinality interaction features and factorize across train+test
    interaction_cols = []
    for col in ['bmi_bin', 'age_bin', 'activity_bin']:
        name = f'{col}_x_fh_fact'
        combined = pd.concat([train[col].astype(str) + '_' + train['family_history_diabetes'].astype(str),
                              test[col].astype(str) + '_' + test['family_history_diabetes'].astype(str)],
                             ignore_index=True)
        codes, _ = pd.factorize(combined)
        # If cardinality too high, drop interaction entirely
        if pd.Series(codes).nunique() > len(codes) // 2:
            continue
        train[name] = codes[:len(train)]
        test[name] = codes[len(train):]
        interaction_cols.append(name)

    NUMS.extend(interaction_cols)
    FEATURES += interaction_cols

    # Memory reduce
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    gc.collect()

    X = train[FEATURES]
    y = train[TARGET]

    return X, y, FEATURES, CATS, NUMS, interaction_cols


# ----------------------
# Training wrappers
# ----------------------

def train_xgb(X, y, test, FEATURES, CATS, TE_cols):
    params = {
        'n_estimators': 20000,
        'learning_rate': 0.01,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc',
        'early_stopping_rounds': 200,
    }

    oof = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
        X_test_fold = test[FEATURES].copy()

        # Target encode interactions (adds TE_ columns)
        if TE_cols:
            te = TargetEncoder(cols_to_encode=TE_cols, aggs=['mean', 'count'], cv=5, smooth='auto')
            X_tr = te.fit_transform(X_tr, y_tr)
            X_val = te.transform(X_val)
            X_test_fold = te.transform(X_test_fold)
            del te

        # Factorize categorical cols so XGBoost can use them if using native categorical mode
        for c in CATS:
            combined = pd.concat([X_tr[c], X_val[c], X_test_fold[c]])
            enc, _ = combined.factorize()
            X_tr[c] = enc[:len(X_tr)].astype('category')
            X_val[c] = enc[len(X_tr):len(X_tr)+len(X_val)].astype('category')
            X_test_fold[c] = enc[len(X_tr)+len(X_val):].astype('category')

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        test_preds += model.predict_proba(X_test_fold)[:, 1] / kf.get_n_splits()
        print(f'XGB Fold {fold+1} AUC: {roc_auc_score(y_val, val_pred):.5f}')
        del model
        gc.collect()

    print('XGB OOF AUC:', roc_auc_score(y, oof))
    return oof, test_preds


def train_lgb(X, y, test, FEATURES, CATS, TE_cols):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 10000,
        'learning_rate': 0.005,
        'num_leaves': 95,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    oof = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
        X_test_fold = test[FEATURES].copy()

        if TE_cols:
            te = TargetEncoder(cols_to_encode=TE_cols, aggs=['mean', 'count'], cv=5, smooth='auto')
            X_tr = te.fit_transform(X_tr, y_tr)
            X_val = te.transform(X_val)
            X_test_fold = te.transform(X_test_fold)
            del te

        for c in CATS:
            combined = pd.concat([X_tr[c], X_val[c], X_test_fold[c]])
            enc, _ = combined.factorize()
            X_tr[c] = enc[:len(X_tr)].astype('category')
            X_val[c] = enc[len(X_tr):len(X_tr)+len(X_val)].astype('category')
            X_test_fold[c] = enc[len(X_tr)+len(X_val):].astype('category')

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', verbose=False)

        val_pred = clf.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        test_preds += clf.predict_proba(X_test_fold)[:, 1] / kf.get_n_splits()
        print(f'LGB Fold {fold+1} AUC: {roc_auc_score(y_val, val_pred):.5f}')
        del clf
        gc.collect()

    print('LGB OOF AUC:', roc_auc_score(y, oof))
    return oof, test_preds


def train_catboost(X, y, test, FEATURES, CATS, TE_cols):
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
        X_test_fold = test[FEATURES].copy()

        if TE_cols:
            te = TargetEncoder(cols_to_encode=TE_cols, aggs=['mean', 'count'], cv=5, smooth='auto')
            X_tr = te.fit_transform(X_tr, y_tr)
            X_val = te.transform(X_val)
            X_test_fold = te.transform(X_test_fold)
            del te

        cat_feats = CATS if CATS else None
        train_pool = Pool(X_tr, y_tr, cat_features=cat_feats)
        val_pool = Pool(X_val, y_val, cat_features=cat_feats)

        cb = CatBoostClassifier(iterations=10000, learning_rate=0.01, depth=6, eval_metric='AUC', random_seed=42, thread_count=-1, verbose=False)
        cb.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)

        val_pred = cb.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        test_preds += cb.predict_proba(X_test_fold)[:, 1] / kf.get_n_splits()
        print(f'CatBoost Fold {fold+1} AUC: {roc_auc_score(y_val, val_pred):.5f}')

        del cb, train_pool, val_pool
        gc.collect()

    print('CatBoost OOF AUC:', roc_auc_score(y, oof))
    return oof, test_preds


# ----------------------
# Main script
# ----------------------
if __name__ == '__main__':
    # Load datasets
    sample = pd.read_csv(SAMPLE_SUB)
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    orig = pd.read_csv(ORIG_CSV)

    X, y, FEATURES, CATS, NUMS, interaction_cols = build_features(train, test, orig)

    # Choose TE columns (interaction features are good TE candidates)
    TE_COLS = interaction_cols

    # Train models and get OOF/test predictions
    oof_xgb, test_xgb = train_xgb(X, y, test, FEATURES, CATS, TE_COLS)
    oof_lgb, test_lgb = train_lgb(X, y, test, FEATURES, CATS, TE_COLS)
    oof_cb, test_cb = train_catboost(X, y, test, FEATURES, CATS, TE_COLS)

    # Simple ensemble: average predictions
    test_ensemble = (test_xgb + test_lgb + test_cb) / 3

    # Save submission (ids kept from sample submission)
    sub = pd.read_csv(SAMPLE_SUB)
    sub[TARGET] = test_ensemble
    sub.to_csv('submission_ensemble.csv', index=False)

    # Save OOFs for stacking/analysis
    oof_df = pd.DataFrame({'id': train['id'], TARGET: y, 'oof_xgb': oof_xgb, 'oof_lgb': oof_lgb, 'oof_cb': oof_cb})
    oof_df.to_csv('oof_predictions_ensemble.csv', index=False)

    print('Saved submission_ensemble.csv and oof_predictions_ensemble.csv')

    # Minimal diagnostic plot
    plt.figure(figsize=(8, 4))
    sns.kdeplot(oof_df['oof_xgb'], label='OOF XGB', fill=True)
    sns.kdeplot(oof_df['oof_lgb'], label='OOF LGB', fill=True)
    plt.legend()
    plt.tight_layout()
    plt.show()
