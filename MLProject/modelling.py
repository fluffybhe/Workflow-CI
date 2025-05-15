import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
from dagshub import dagshub
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Configure MLflow tracking with DagsHub"""
    try:
        # Initialize DagsHub connection
        dagshub.init(
            repo_owner="fluffybhe",
            repo_name="Eksperimen_SML_Febhe",
            mlflow=True,
            token=os.getenv("DAGSHUB_TOKEN")
        )
        
        # Set tracking URI
        mlflow.set_tracking_uri(f"https://dagshub.com/fluffybhe/Eksperimen_SML_Febhe.mlflow")
        mlflow.set_experiment("California-Housing")
        
        logger.info("✅ Successfully configured MLflow with DagsHub")
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        
    except Exception as e:
        logger.error(f"❌ Failed to setup DagsHub: {str(e)}")
        logger.info("⚠️ Falling back to local MLflow tracking")
        mlflow.set_tracking_uri("file:///tmp/mlruns")

def load_data():
    """Load and preprocess data"""
    try:
        data_path = os.path.join("MLProject", "processed_california_housing.csv")
        logger.info(f"📂 Loading data from: {data_path}")
        
        data = pd.read_csv(data_path)
        logger.info(f"✅ Data loaded. Shape: {data.shape}")
        
        # Preprocessing
        data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())
        X = data.drop("median_house_value", axis=1)
        y = data["median_house_value"]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {str(e)}")
        raise

def train_model(X_train, X_test, y_train, y_test):
    """Train and log model with MLflow"""
    try:
        mlflow.sklearn.autolog()
        
        with mlflow.start_run(run_name="RF_Production"):
            logger.info("🏋️ Training Random Forest model...")
            
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mlflow.log_metric("MAE", mae)
            
            # Save model
            model_path = "random_forest_model.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            
            logger.info(f"🎉 Training complete! MAE: {mae:.2f}")
            
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}")
        raise

def main():
    """Main execution flow"""
    logger.info("🚀 Starting ML pipeline")
    
    # 1. Setup MLflow
    setup_mlflow()
    
    # 2. Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # 3. Train model
    train_model(X_train, X_test, y_train, y_test)
    
    logger.info("🏁 Pipeline completed successfully")

if __name__ == "__main__":
    main()
