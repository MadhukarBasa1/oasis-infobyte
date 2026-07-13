# Car Price Prediction with Machine Learning

This project trains a regression model on your current car dataset to predict selling prices from the available car attributes.

## What the project does
- Reads the current dataset from data/car_data.csv.
- Trains a regression model using the car name, year, present price, driven kilometers, fuel type, seller type, transmission, and owner count.
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
python src/predict.py --car-name ritz --year 2014 --present-price 5.59 --driven-kms 27000 --fuel-type Petrol --selling-type Dealer --transmission Manual --owner 0
```

## Project structure
- src/train_model.py: dataset loading, training, and model saving
- src/predict.py: command-line prediction interface
- data/car_data.csv: current car dataset
- models/car_price_model.joblib: trained model
