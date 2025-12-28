from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
import pandas as pd


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Simple, commented target encoder.

    - Encodes categorical columns by aggregating the target per category.
    - Supports multiple aggregations (e.g., 'mean', 'std').
    - `fit_transform` uses internal CV to avoid target leakage; smoothing
      for the 'mean' aggregation is applied per-fold only.

    Notes / caveats:
    - Smoothing is applied only for 'mean' inside `fit_transform`; `transform`
      uses the raw mappings learned in `fit` (no smoothing persisted).
    - Intended for numeric targets (regression) or binary targets encoded as
      0/1. Not directly suitable for multiclass targets without modification.
    - For time-series or grouped data, pass a suitable splitter (not supported
      by this class yet). KFold is used by default.
    """

    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        # user parameters
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original

        # learned attributes populated in `fit`
        self.mappings_ = {}      # per-column, per-agg mapping series
        self.global_stats_ = {}  # global aggregated target for each agg

    def fit(self, X, y):
        """Learn mappings from the full dataset.

        These mappings are used by `transform` to encode new data. Note that no
        fold-based smoothing is applied here — mappings are the raw group aggs.
        """
        temp_df = X.copy()
        temp_df['target'] = y

        # global statistic for each aggregation (used for unseen categories)
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)

        # store category -> aggregated target mappings for each column and agg
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping

        return self

    def transform(self, X):
        """Apply learned mappings to `X`.

        - New columns are named `TE_<col>_<agg>`.
        - Categories not seen in `fit` are filled with the corresponding
          global statistic from `self.global_stats_`.
        - IMPORTANT: If you relied on fold-wise smoothing in `fit_transform`,
          `transform` will not apply that smoothing because `fit` stores raw
          group aggregates. Persisting smoothed values would require computing
          and storing them in `fit`.
        """
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                # fill unseen categories with global stat
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)

        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)

        return X_transformed

    def fit_transform(self, X, y):
        """Fit and transform using internal CV to prevent leakage.

        For each fold, mappings are learned on the training part and applied to
        the validation part. This prevents using the validation target to
        encode its own rows.

        Smoothing: only applied for the 'mean' aggregation. When `smooth='auto'`
        a simple empirical-Bayes heuristic is used (ratio of within-group and
        between-group variances). The smoothing is computed per-fold and used
        for that fold's encodings; these smoothed values are not written back
        to `self.mappings_` (so `transform` remains unsmoothed).
        """
        # fit on the whole data to populate mappings_ and global_stats_
        # (transform will use these mappings for new/unseen data)
        self.fit(X, y)

        # DataFrame to collect encoded features for all validation folds
        encoded_features = pd.DataFrame(index=X.index)

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                # compute group-based statistics on fold's training data
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'

                    fold_global_stat = y_train.agg(agg_func)
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    if agg_func == 'mean':
                        # apply smoothing to the mean only
                        counts = temp_df_train.groupby(col)['target'].count()

                        m = self.smooth
                        if self.smooth == 'auto':
                            # empirical-Bayes style heuristic (may be unstable on tiny data)
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0

                        # (counts * group_mean + m * global_mean) / (counts + m)
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)

                    # fill missing (unseen in training fold) with fold global stat
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        # attach encoded features to a copy of the original DataFrame
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]

        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)

        return X_transformed