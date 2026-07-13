import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "car_data.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "car_price_model.joblib")

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def resolve_data_path():
    preferred_path = os.path.join(ROOT_DIR, "data", "car_data.csv")
    legacy_path = os.path.join(ROOT_DIR, "data", "car_price_data.csv")
    if os.path.exists(preferred_path):
        return preferred_path
    if os.path.exists(legacy_path):
        return legacy_path
    return preferred_path


def get_feature_columns():
    return ["Car_Name", "Year", "Present_Price", "Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner"]


def load_training_data(path=None):
    data_path = path or resolve_data_path()
    df = pd.read_csv(data_path)

    df = df.copy()
    if "Selling_Price" in df.columns:
        df["price"] = df["Selling_Price"]
    elif "price" in df.columns:
        df["price"] = df["price"]
    else:
        raise ValueError("The dataset must contain a 'Selling_Price' or 'price' column for training.")

    if "Selling_Price" not in df.columns:
        df["Selling_Price"] = df["price"]

    required_columns = get_feature_columns() + ["price"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"The dataset is missing required columns: {missing_columns}")

    return df[[*get_feature_columns(), "price", "Selling_Price"]].copy()


def build_pipeline():
    categorical_features = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
    numeric_features = ["Year", "Present_Price", "Driven_kms", "Owner"]

    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
        ]
    )

    model = RandomForestRegressor(n_estimators=250, random_state=42)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("model", model),
        ]
    )
    return pipeline


def train_model():
    df = load_training_data(resolve_data_path())
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    joblib.dump(pipeline, MODEL_PATH)

    print(f"Dataset used: {resolve_data_path()}")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Mean absolute error: ₹{mae:,.0f}")
    print(f"R2 score: {r2:.3f}")


def load_model():
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train_model()
