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
dagshub.init(
    repo_owner='fluffybhe',
    repo_name='Eksperimen_SML_Febhe',
    mlflow=True
)
mlflow.set_tracking_uri("https://dagshub.com/fluffybhe/Eksperimen_SML_Febhe.mlflow")
mlflow.set_experiment("california_housing_experiment")

# Load data
data_path = "MLProject/processed_california_housing.csv"
data = pd.read_csv(data_path)

# Preprocessing
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean()).astype('float64')

# Pisahkan fitur dan target
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Aktifkan autologging MLflow
mlflow.sklearn.autolog()

# Mulai eksperimen
with mlflow.start_run(run_name="random_forest_complete") as run:
    # Inisialisasi dan latih model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Hitung dan log metrik MAE
    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metric("mae", mae)

    # Simpan model eksplisit (opsional, karena autolog sudah menangani ini)
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Training selesai, MAE: {mae:.2f}")
    print(f"Run ID: {run.info.run_id}")


