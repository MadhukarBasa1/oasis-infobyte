import argparse
import os
import pandas as pd

from train_model import MODEL_PATH, load_model


def predict_price(car_name, year, present_price, driven_kms, fuel_type, selling_type, transmission, owner):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train the model first by running python src/train_model.py")

    model = load_model()
    input_df = pd.DataFrame(
        [
            {
                "Car_Name": car_name,
                "Year": year,
                "Present_Price": present_price,
                "Driven_kms": driven_kms,
                "Fuel_Type": fuel_type,
                "Selling_type": selling_type,
                "Transmission": transmission,
                "Owner": owner,
            }
        ]
    )
    predicted_price = model.predict(input_df)[0]
    print(f"Predicted car price: ₹{predicted_price:,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a car selling price")
    parser.add_argument("--car-name", default="ritz")
    parser.add_argument("--year", type=int, default=2014)
    parser.add_argument("--present-price", type=float, default=5.59)
    parser.add_argument("--driven-kms", type=int, default=27000)
    parser.add_argument("--fuel-type", default="Petrol")
    parser.add_argument("--selling-type", default="Dealer")
    parser.add_argument("--transmission", default="Manual")
    parser.add_argument("--owner", type=int, default=0)
    args = parser.parse_args()

    predict_price(
        car_name=args.car_name,
        year=args.year,
        present_price=args.present_price,
        driven_kms=args.driven_kms,
        fuel_type=args.fuel_type,
        selling_type=args.selling_type,
        transmission=args.transmission,
        owner=args.owner,
    )
