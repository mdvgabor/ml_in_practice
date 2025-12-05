import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 1

def load_data():
    white = pd.read_csv("winequality-white.csv", sep=";")
    red = pd.read_csv("winequality-red.csv", sep=";")

    white["type"] = "white"
    red["type"] = "red"

    return pd.concat([white, red], ignore_index=True)

def train_classification_model(df):
    X = df.drop(["type", "quality"], axis=1)
    y = df["type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), X.columns)
        ]
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred)
    }

    # Save the model
    joblib.dump(model, "models/classification_rf.joblib")

    return model, metrics

if __name__ == "__main__":
    df = load_data()
    model, metrics = train_classification_model(df)
    print("Model retrained successfully.")
    print(metrics)