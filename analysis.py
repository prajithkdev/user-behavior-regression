"""
Predicts product price from user-behavior and catalog signals in the
REES46 eCommerce events dataset (event_type, category, brand, time-of-day).

This is a from-scratch Python reimplementation of an original R group
assignment. The original R script tried to use `user_id` (millions of
unique values) and `category_id`/`category_code` as raw high-cardinality
factors in a linear model — a setup that, per the original report, produced
unfilled results and runtime errors ("object 'inDataTrainingData' not
found", referencing a variable from an unrelated dataset). This version
drops `user_id` entirely (a near-unique identifier isn't a generalizable
predictor of price) and uses the top-N most frequent categories/brands
instead of the full raw cardinality, so the model can actually train.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
SAMPLE_SIZE = 20000
TOP_N_CATEGORIES = 100
TOP_N_BRANDS = 100

print("Loading data...")
df = pd.read_csv(
    "C:/Masters/DATASETS/eventsEcommerce.csv",
    usecols=["event_time", "event_type", "category_code", "brand", "price"],
)
print(f"Full dataset: {len(df):,} rows")

df = df.dropna(subset=["category_code", "brand", "price"])
df = df[df["price"] > 0]
print(f"After dropping missing category/brand/price: {len(df):,} rows")

df["hour"] = pd.to_datetime(df["event_time"]).dt.hour

top_categories = df["category_code"].value_counts().nlargest(TOP_N_CATEGORIES).index
top_brands = df["brand"].value_counts().nlargest(TOP_N_BRANDS).index
df["category_code"] = df["category_code"].where(df["category_code"].isin(top_categories), "other")
df["brand"] = df["brand"].where(df["brand"].isin(top_brands), "other")

sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)

X = sample[["event_type", "category_code", "brand", "hour"]]
y = sample["price"]

categorical_features = ["event_type", "category_code", "brand"]
preprocess = ColumnTransformer(
    [("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough",
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

models = {
    "Linear Regression": LinearRegression(),
    "Lasso (alpha=1.0)": Lasso(alpha=1.0, max_iter=10000),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
}

results = []
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, rmse, r2))
    print(f"{name}: RMSE={rmse:.2f}, R2={r2:.4f}")

print("\n| Model | RMSE | R² |")
print("|---|---|---|")
for name, rmse, r2 in results:
    print(f"| {name} | {rmse:.2f} | {r2:.4f} |")
