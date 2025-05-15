import os
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mlflow
from dagshub.auth import token_auth
from dagshub.common import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Initialize connection to DagsHub"""
    try:
        # Set token first
        token = os.getenv("DAGSHUB_TOKEN")
        if not token:
            raise ValueError("Token tidak ditemukan!")
            
        token_auth(token)  # Authenticate
        
        # Manual MLflow setup (lebih stabil)
        mlflow.set_tracking_uri(
            f"https://{token}@dagshub.com/fluffybhe/Eksperimen_SML_Febhe.mlflow"
        )
        mlflow.set_experiment("California-Housing")
        
        logger.info("MLflow siap di DagsHub!")
    except Exception as e:
        logger.warning(f"Gagal konek ke DagsHub: {e}")
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        logger.info("Pakai MLflow lokal saja")

def main():
    logger.info("Memulai pipeline...")
    
    # 1. Setup MLflow
    setup_mlflow()
    
    # 2. Load data
    data = pd.read_csv("MLProject/processed_california_housing.csv")
    data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train model
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mlflow.log_metric("mae", mae)
        logger.info(f"MAE: {mae:.2f}")

if __name__ == "__main__":
    main()
