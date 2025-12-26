import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os

# --- Configuration ---
# File paths for OOF predictions (Train set for Meta-Model)
OOF_FILES = {
    'xgb': 'oof_predictions.csv',
    'lgbm': 'oof_lgbm.csv',
    'cb': 'oof_catboost.csv'
}

# File paths for Test predictions (Test set for Meta-Model)
TEST_FILES = {
    'xgb': 'submission_robust_te_withid.csv',
    'lgbm': 'submission_lgbm.csv',
    'cb': 'submission_catboost.csv'
}

TARGET_COL = 'diagnosed_diabetes'
ID_COL = 'id'
OUTPUT_FILE = 'submission_stacking.csv'

def load_and_merge(files_dict, is_train=True):
    """
    Loads prediction files and merges them into a single DataFrame.
    """
    print(f"Loading {'Train' if is_train else 'Test'} data...")
    merged_df = None
    
    for model_name, file_path in files_dict.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        print(f"  Loading {model_name}: {file_path}")
        df = pd.read_csv(file_path)
        
        # Standardize column names if necessary (assuming 'pred' or target col name)
        # We need specific prediction column. 
        # Based on previous analysis:
        # OOF files have: id, target, pred
        # Submission files have: id, target (as probability)
        
        pred_col = 'pred' if is_train else TARGET_COL
        
        # Rename prediction column to model name
        df = df.rename(columns={pred_col: f'pred_{model_name}'})
        
        if merged_df is None:
            # First dataframe initializes the merge
            cols_to_keep = [ID_COL, f'pred_{model_name}']
            if is_train and TARGET_COL in df.columns:
                cols_to_keep.append(TARGET_COL)
            merged_df = df[cols_to_keep]
        else:
            # Merge subsequent dataframes on ID
            merged_df = merged_df.merge(df[[ID_COL, f'pred_{model_name}']], on=ID_COL, how='left')
            
    return merged_df

def main():
    # 1. Prepare Level 1 Train Data (Meta-Features)
    train_meta = load_and_merge(OOF_FILES, is_train=True)
    
    # 2. Prepare Level 1 Test Data
    test_meta = load_and_merge(TEST_FILES, is_train=False)
    
    feature_cols = [f'pred_{m}' for m in OOF_FILES.keys()]
    print(f"\nMeta-Features: {feature_cols}")
    
    # 3. Train Meta-Model
    print("\nTraining Meta-Model (Logistic Regression)...")
    X_train = train_meta[feature_cols]
    y_train = train_meta[TARGET_COL]
    
    meta_model = LogisticRegression()
    meta_model.fit(X_train, y_train)
    
    # Validation on OOF (Sanity Check - reusing training data logic but helpful for checking fit)
    # Ideally we'd cross-validate this too, but for simple stacking on full OOF, simple fit is standard step
    train_preds = meta_model.predict_proba(X_train)[:, 1]
    auc_score = roc_auc_score(y_train, train_preds)
    print(f"Meta-Model OOF AUC Score (approx): {auc_score:.5f}")
    
    print("\nModel Coefficients:")
    for feature, coef in zip(feature_cols, meta_model.coef_[0]):
        print(f"  {feature}: {coef:.4f}")
        
    # 4. Generate Final Predictions
    print("\nGenerating final predictions on Test set...")
    X_test = test_meta[feature_cols]
    test_preds = meta_model.predict_proba(X_test)[:, 1]
    
    # 5. Create Submission File
    submission = pd.DataFrame({
        ID_COL: test_meta[ID_COL],
        TARGET_COL: test_preds
    })
    
    print(f"\nSaving submission to {OUTPUT_FILE}...")
    submission.to_csv(OUTPUT_FILE, index=False)
    print("Done!")
    print(f"Submission shape: {submission.shape}")
    print(submission.head())

if __name__ == "__main__":
    main()
