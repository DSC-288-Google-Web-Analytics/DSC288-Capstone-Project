import polars as pl
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def safe_json(x):
    try:
        return json.loads(x)
    except:
        return {}


def column_split(df):
    device = pd.json_normalize(df["device"])
    device = device[["isMobile", "deviceCategory"]]

    geo = pd.json_normalize(df["geoNetwork"])
    geo = geo[["subContinent", "country"]]

    totals = pd.json_normalize(df["totals"])
    totals = totals.rename(columns={"hits": "hit"})

    keep_totals = [
        "visits",
        "hit",
        "pageviews",
        "transactions",
        "transactionRevenue"
    ]
    totals = totals[keep_totals].apply(pd.to_numeric, errors="coerce")

    df_new = pd.concat([
        df.drop(columns=["device", "geoNetwork", "totals", "trafficSource"]),
        device, geo, totals
    ], axis=1)

    return df_new


def is_revenue(dataset):
    df = dataset.copy()
    df['transactionRevenue'] = pd.to_numeric(df['transactionRevenue'], errors='coerce').fillna(0)
    df['transactions'] = pd.to_numeric(df['transactions'], errors='coerce').fillna(0)
    df['has_revenue'] = (df['transactionRevenue'] > 0).astype(int)
    return df


def type_change(df):
    df['visitNumber'] = pd.to_numeric(df['visitNumber'], errors='coerce')
    df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit='s')

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TwoStageRevenueModel:
    def __init__(self, model_type='lgb', clf_params=None, reg_params=None, threshold=0.5):
        if model_type not in ('lgb', 'xgb'):
            raise ValueError(f"model_type must be 'lgb' or 'xgb', got '{model_type}'")
        self.model_type = model_type
        self.threshold = threshold

        if model_type == 'lgb':
            self.clf_params = clf_params or {
                'objective': 'binary', 'boosting_type': 'gbdt',
                'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31,
                'max_depth': -1, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            self.reg_params = reg_params or {
                'objective': 'regression', 'n_estimators': 500, 'learning_rate': 0.05,
                'num_leaves': 64, 'max_depth': 6, 'min_child_samples': 100,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1,
                'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            self.clf = lgb.LGBMClassifier(**self.clf_params)
            self.reg = lgb.LGBMRegressor(**self.reg_params)
        else:
            self.clf_params = clf_params or {
                'objective': 'binary:logistic', 'n_estimators': 300,
                'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8,
                'colsample_bytree': 0.8, 'scale_pos_weight': 10,
                'enable_categorical': True, 'random_state': 42, 'n_jobs': -1
            }
            self.reg_params = reg_params or {
                'objective': 'reg:squarederror', 'n_estimators': 500,
                'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 100,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1,
                'reg_lambda': 1.0, 'enable_categorical': True, 'random_state': 42, 'n_jobs': -1
            }
            self.clf = xgb.XGBClassifier(**self.clf_params)
            self.reg = xgb.XGBRegressor(**self.reg_params)

    def fit(self, X, y_cls, y_reg):
        self.clf.fit(X, y_cls)
        buyer_mask = y_cls == 1
        self.reg.fit(X[buyer_mask], y_reg[buyer_mask])

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def predict_expected_revenue(self, X):
        p_buy = self.predict_proba(X)
        rev = self.reg.predict(X)
        return p_buy * rev  

    def evaluate_classification(self, X, y):
        return roc_auc_score(y, self.predict_proba(X))

    def evaluate_regression(self, X, y):
        pred = self.predict_expected_revenue(X)
        return float(np.sqrt(mean_squared_error(y, pred)))


# ---------------------------------------------------------------------------
# Data processor  (fixed typo: DataProcsser → DataProcessor)
# ---------------------------------------------------------------------------

class DataProcessor:
    def __init__(self, path_file=None):
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.y_train_reg = self.y_val_reg = self.y_test_reg = None
        if path_file is None:
            self.data = None
        else:
            self.data = self.get_data(path_file)
            self.split_data()

    def get_data(self, file_path, keep_cols=None, slice_iters=1_000_000):
        if keep_cols is None:
            keep_cols = ['fullVisitorId', 'channelGrouping', 'device', 'geoNetwork',
                         'trafficSource', 'visitNumber', 'visitStartTime', 'date', 'totals']

        df_pl = pl.scan_csv(file_path, schema_overrides={'fullVisitorId': pl.Utf8})
        features = list(set(keep_cols) & set(df_pl.collect_schema().names()))
        if not features:
            print('No matching features!')
            return

        df_pl = df_pl.select(features)
        all_chunks, count = [], 0
        for chunk in df_pl.collect(engine='streaming').iter_slices(slice_iters):
            chunk_pd = chunk.to_pandas()
            for col in ['device', 'geoNetwork', 'totals']:
                if col in chunk_pd.columns:
                    chunk_pd[col] = chunk_pd[col].apply(safe_json)
            chunk_pd = column_split(chunk_pd)
            count += len(chunk_pd)
            all_chunks.append(chunk_pd)

        data = pd.concat(all_chunks, axis=0)
        data = is_revenue(data)
        data = type_change(data)
        print(f'Loaded {data.shape[0]:,} rows, {data.shape[1]} columns')
        return data

    def split_data(self, features=None):
        if features is None:
            features = ['channelGrouping', 'visitNumber', 'isMobile', 'deviceCategory',
                        'subContinent', 'country', 'year', 'month', 'day', 'hit', 'pageviews']

        users = self.data['fullVisitorId'].unique()
        np.random.shuffle(users)
        n = len(users)
        train_users = users[:int(0.8 * n)]
        val_users   = users[int(0.8 * n):int(0.9 * n)]
        test_users  = users[int(0.9 * n):]

        train_df = self.data[self.data.fullVisitorId.isin(train_users)]
        val_df   = self.data[self.data.fullVisitorId.isin(val_users)]
        test_df  = self.data[self.data.fullVisitorId.isin(test_users)]

        features = list(set(features) & set(self.data.columns))
        self.X_train = train_df[features].copy()
        self.X_val   = val_df[features].copy()
        self.X_test  = test_df[features].copy()

        self.y_train = train_df['has_revenue']
        self.y_val   = val_df['has_revenue']
        self.y_test  = test_df['has_revenue']
        cat_cols = ['channelGrouping', 'deviceCategory', 'subContinent', 'country']
        for col in cat_cols:
            for split in [self.X_train, self.X_val, self.X_test]:
                if col in split.columns:
                    split[col] = split[col].astype('category')
        for split in [self.X_train, self.X_val, self.X_test]:
            if 'isMobile' in split.columns:
                split['isMobile'] = split['isMobile'].astype(float).fillna(0).astype(int)

        self.y_train_reg = np.log1p(train_df['transactionRevenue'])
        self.y_val_reg   = np.log1p(val_df['transactionRevenue'])
        self.y_test_reg  = np.log1p(test_df['transactionRevenue'])

        print(f'Train: {len(self.X_train):,} rows | Val: {len(self.X_val):,} | Test: {len(self.X_test):,}')
        print(f'Train positive rate: {self.y_train.mean():.4%}')