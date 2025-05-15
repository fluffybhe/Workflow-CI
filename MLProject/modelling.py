import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
import dagshub

# Inisialisasi DagsHub + MLflow
dagshub.init(repo_owner='fluffybhe', repo_name='Eksperimen_SML_Febhe', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/fluffybhe/Eksperimen_SML_Febhe.mlflow")
mlflow.set_experiment("california_housing_experiment")

data_path = "MLProject/processed_california_housing.csv"
data = pd.read_csv(data_path)
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean()).astype('float64')

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="random_forest_complete"):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metric("mae", mae)

    print(f"Training selesai, MAE: {mae:.2f}")
