# Car Price Prediction with Machine Learning

This project demonstrates a simple end-to-end regression workflow for predicting car prices using a machine learning model.

## What the project does
- Generates a synthetic car dataset with features such as brand, year, mileage, horsepower, engine size, fuel type, transmission, and owner count.
- Trains a regression model to estimate the price of a car.
- Saves the trained model and allows you to make new predictions from the command line.

## Setup
Install the required packages:

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python src/train_model.py
```

## Predict a car price

```bash
python src/predict.py --brand Toyota --year 2022 --mileage 18000 --horsepower 180 --engine-size 2.0 --fuel-type Petrol --transmission Automatic --owner-count 1
```

## Project structure
- src/train_model.py: dataset generation, training, and model saving
- src/predict.py: command-line prediction interface
- data/car_price_data.csv: generated dataset
- models/car_price_model.joblib: trained model
