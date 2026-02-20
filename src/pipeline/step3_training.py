import joblib
import os
# Change evaluate_reg to evaluate_regression
from utils.model_utils import save_artifact, get_price_bins, evaluate_regression 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def run_training_pipeline():
    # 1. Load data from step 2
    X_train = joblib.load('data/processed/X_final.joblib') 
    y_train = joblib.load('data/processed/y_final.joblib')

    print("--- Training Regression Model ---")
    reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    save_artifact(reg, "price_regressor")
    
    # Optional: If you want to see training metrics immediately
    # metrics = evaluate_regression(reg, X_train, y_train)
    # print(f"Training MAE: €{metrics['MAE']:.2f}")

    print("--- Training Classification Model ---")
    y_class = get_price_bins(y_train)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_class)
    save_artifact(clf, "price_classifier")
    
    print("🚀 Training Complete: Regressor & Classifier saved.")

if __name__ == "__main__":
    run_training_pipeline()