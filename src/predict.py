import argparse
import os
import pandas as pd

from train_model import MODEL_PATH, load_model


def predict_price(brand, year, mileage, horsepower, engine_size, fuel_type, transmission, owner_count):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train the model first by running python src/train_model.py")

    model = load_model()
    input_df = pd.DataFrame(
        [
            {
                "brand": brand,
                "year": year,
                "mileage": mileage,
                "horsepower": horsepower,
                "engine_size": engine_size,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "owner_count": owner_count,
            }
        ]
    )
    predicted_price = model.predict(input_df)[0]
    print(f"Predicted car price: ₹{predicted_price:,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a car price")
    parser.add_argument("--brand", default="Toyota")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--mileage", type=int, default=18000)
    parser.add_argument("--horsepower", type=int, default=180)
    parser.add_argument("--engine-size", type=float, default=2.0)
    parser.add_argument("--fuel-type", default="Petrol")
    parser.add_argument("--transmission", default="Automatic")
    parser.add_argument("--owner-count", type=int, default=1)
    args = parser.parse_args()

    predict_price(
        brand=args.brand,
        year=args.year,
        mileage=args.mileage,
        horsepower=args.horsepower,
        engine_size=args.engine_size,
        fuel_type=args.fuel_type,
        transmission=args.transmission,
        owner_count=args.owner_count,
    )
