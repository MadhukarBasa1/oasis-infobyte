import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "car_price_data.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "car_price_model.joblib")

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def build_dataset(n_rows=300, seed=42):
    rng = np.random.default_rng(seed)
    brands = ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Tata", "Hyundai"]
    fuel_types = ["Petrol", "Diesel", "Electric"]
    transmissions = ["Manual", "Automatic"]

    data = []
    for _ in range(n_rows):
        brand = rng.choice(brands)
        year = int(rng.integers(2015, 2025))
        mileage = int(rng.integers(5000, 120000))
        horsepower = int(rng.integers(80, 320))
        engine_size = round(float(rng.uniform(1.0, 3.2)), 1)
        fuel_type = rng.choice(fuel_types)
        transmission = rng.choice(transmissions)
        owner_count = int(rng.integers(1, 5))

        brand_multiplier = {
            "Toyota": 1.02,
            "Honda": 1.04,
            "Ford": 0.95,
            "BMW": 1.18,
            "Mercedes": 1.22,
            "Tata": 0.90,
            "Hyundai": 0.93,
        }[brand]
        fuel_bonus = {"Petrol": 10000, "Diesel": 18000, "Electric": 22000}[fuel_type]
        transmission_bonus = 12000 if transmission == "Automatic" else 0
        owner_penalty = owner_count * 15000

        price = (
            500000
            + (year - 2015) * 50000
            - mileage * 0.8
            + horsepower * 1500
            + engine_size * 70000
            + fuel_bonus
            + transmission_bonus
            - owner_penalty
        )
        price = int(round(price * brand_multiplier + rng.normal(0, 25000)))
        price = max(180000, price)

        data.append(
            {
                "brand": brand,
                "year": year,
                "mileage": mileage,
                "horsepower": horsepower,
                "engine_size": engine_size,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "owner_count": owner_count,
                "price": price,
            }
        )

    return pd.DataFrame(data)


def build_pipeline():
    categorical_features = ["brand", "fuel_type", "transmission"]
    numeric_features = ["year", "mileage", "horsepower", "engine_size", "owner_count"]

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
    if not os.path.exists(DATA_PATH):
        df = build_dataset()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    joblib.dump(pipeline, MODEL_PATH)

    print(f"Dataset saved at: {DATA_PATH}")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Mean absolute error: ₹{mae:,.0f}")
    print(f"R2 score: {r2:.3f}")


def load_model():
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train_model()
