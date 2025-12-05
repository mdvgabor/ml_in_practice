import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 1

def load_data():
    white = pd.read_csv("winequality-white.csv", sep=";")
    red = pd.read_csv("winequality-red.csv", sep=";")

    white["type"] = "white"
    red["type"] = "red"

    df = pd.concat([white, red], ignore_index=True)
    df["type"] = df["type"].astype("category")
    return df

def train_regression_model(df):
    X = df.drop("quality", axis=1)
    y = df["quality"]

    numeric_features = X.drop(columns=["type"]).columns
    categorical_features = ["type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )

    rf_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE))
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 10]
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred),
    }

    joblib.dump(best_model, "models/regression_rf.joblib")

    return best_model, metrics, y_test, y_pred

if __name__ == "__main__":
    df = load_data()
    model, metrics, _, _ = train_regression_model(df)
    print(metrics)
    print("x")