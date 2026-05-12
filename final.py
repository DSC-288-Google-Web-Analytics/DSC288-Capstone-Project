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
    def __init__(self,
                 clf_params=None,
                 reg_params=None,
                 threshold=0.5):

        self.clf_params = clf_params or {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }

        self.reg_params = reg_params or {
            "objective": "regression",
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 64,
            "max_depth": 8,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1
        }

        self.threshold = threshold
        self.clf = lgb.LGBMClassifier(**self.clf_params)
        self.reg = lgb.LGBMRegressor(**self.reg_params)

    # -------------------------
    # Stage 1: classification
    # -------------------------
    def fit_classifier(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    # -------------------------
    # Stage 2: regression
    # -------------------------
    def fit_regressor(self, X, y):
        self.reg.fit(X, y)

    def predict_revenue(self, X):
        # FIX: was self.clf.predict(X) — must use the regressor, not the classifier
        return self.reg.predict(X)

    # -------------------------
    # Full training
    # -------------------------
    def fit(self, X, y_cls, y_reg):
        self.fit_classifier(X, y_cls)
        self.fit_regressor(X, y_reg)

    # -------------------------
    # Final prediction
    # -------------------------
    def predict_expected_revenue(self, X):
        p_buy = self.predict_proba(X)
        rev = self.reg.predict(X)
        return p_buy * np.expm1(rev)

    # -------------------------
    # Optional hard threshold version
    # -------------------------
    def predict_threshold_revenue(self, X):
        p_buy = self.predict_proba(X)
        rev = self.predict_revenue(X)
        return (p_buy > self.threshold) * np.expm1(rev)

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate_classification(self, X, y):
        p = self.predict_proba(X)
        return roc_auc_score(y, p)

    def evaluate_regression(self, X, y):
        pred = self.predict_expected_revenue(X)
        return np.sqrt(mean_squared_error(y, pred))


# ---------------------------------------------------------------------------
# Data processor  (fixed typo: DataProcsser → DataProcessor)
# ---------------------------------------------------------------------------

class DataProcessor:
    def __init__(self, path_file=None):
    
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.y_train_reg = None
        self.y_val_reg = None
        self.y_test_reg = None
        if path_file is None:
            self.data = None
        else:
            self.data = self.get_data(path_file)
            # FIX: only call split_data when data was actually loaded
            self.split_data()

    # -------------------------
    # Processing data
    # -------------------------
    def get_data(self,
                 file_path,
                 keep_cols=[
                     "fullVisitorId",
                     "channelGrouping",
                     "device",
                     "geoNetwork",
                     "trafficSource",
                     "visitNumber",
                     "visitStartTime",
                     "date",
                     "totals"
                 ],
                 slice_iters=1000000):

        df = pl.scan_csv(file_path,
                         schema_overrides={"fullVisitorId": pl.Utf8})

        features = list(set(keep_cols) & set(df.collect_schema().names()))
        if len(features) == 0:
            print("No Features can used!!!!")
            return

        df = df.select(features)

        all_chunks = []
        count = 0
        for chunk in df.collect(engine="streaming").iter_slices(slice_iters):
            df_chunk = chunk.to_pandas()
            df_chunk["device"] = df_chunk["device"].apply(safe_json)
            df_chunk["geoNetwork"] = df_chunk["geoNetwork"].apply(safe_json)
            df_chunk["totals"] = df_chunk["totals"].apply(safe_json)
            df_chunk = column_split(df_chunk)
            count += len(df_chunk)
            all_chunks.append(df_chunk)

        try:
            data = pd.concat(all_chunks, axis=0)
            if data.shape[0] != count:
                raise ValueError(
                    f"Row count mismatch: expected {count} rows but got {data.shape[0]}. "
                    "Some chunks may have been lost during concatenation."
                )
        except ValueError as e:
            print(f"[ERROR] {e}")
            raise SystemExit(1)

        data = is_revenue(data)
        data = type_change(data)

        return data

    # -------------------------
    # Split data
    # -------------------------
    def split_data(self, features=[
        "channelGrouping",
        "visitNumber",
        "isMobile",
        "deviceCategory",
        "subContinent",
        "country",
        "year",
        "month",
        "day"
    ]):
        users = self.data["fullVisitorId"].unique()
        np.random.shuffle(users)

        train_size = int(0.8 * len(users))
        val_size = int(0.1 * len(users))

        train_users = users[:train_size]
        val_users = users[train_size:train_size + val_size]
        test_users = users[train_size + val_size:]

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

        cat_cols = [
            "channelGrouping",
            "isMobile",
            "deviceCategory",
            "subContinent",
            "country"
        ]
        for col in cat_cols:
            self.X_train[col] = self.X_train[col].astype("category")
            self.X_val[col]   = self.X_val[col].astype("category")
            self.X_test[col]  = self.X_test[col].astype("category")

        self.y_train_reg = np.log1p(train_df["transactionRevenue"])
        self.y_val_reg   = np.log1p(val_df["transactionRevenue"])
        self.y_test_reg  = np.log1p(test_df["transactionRevenue"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = DataProcessor('./data/train_v2.csv')

    model = TwoStageRevenueModel()
    model.fit_classifier(df.X_train, df.y_train)

    print("Validation AUC:", model.evaluate_classification(df.X_val, df.y_val))
    print("Test AUC:",       model.evaluate_classification(df.X_test, df.y_test))

    model.fit_regressor(df.X_train, df.y_train_reg)

    print("Val RMSE: ",  model.evaluate_regression(df.X_val,  df.y_val_reg))
    print("Test RMSE: ", model.evaluate_regression(df.X_test, df.y_test_reg))

    print(model.predict_revenue(df.X_val))
