import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
from dagshub import dagshub
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Initialize DagsHub with token-only auth
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise ValueError("❌ DAGSHUB_TOKEN not set in environment variables!")

    logger.info("🔗 Connecting to DagsHub...")
    dagshub.init(
        repo_owner="fluffybhe",
        repo_name="Eksperimen_SML_Febhe",  # Nama repo di DagsHub
        mlflow=True,
        token=dagshub_token
    )
    
    # 2. Explicit MLflow setup
    mlflow.set_tracking_uri(f"https://dagshub.com/fluffybhe/Eksperimen_SML_Febhe.mlflow")
    mlflow.set_experiment("California-Housing")

    # 3. Load data (pastikan path sesuai struktur GitHub repo)
    data_path = os.path.join("MLProject", "processed_california_housing.csv")
    logger.info(f"📂 Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    
    # 4. Preprocessing & modeling
    data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. MLflow logging
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="RF-Production"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mlflow.log_metric("MAE", mae)
        logger.info(f"✅ Model trained! MAE: {mae:.2f}")

if __name__ == "__main__":
    main()
