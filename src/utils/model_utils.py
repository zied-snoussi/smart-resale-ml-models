import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

def evaluate_regression(model, X, y):
    """Calculates MAE and R2 for regression models."""
    preds = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, preds),
        "R2": r2_score(y, preds)
    }

def evaluate_classification(model, X, y):
    """Calculates accuracy for classification models."""
    preds = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, preds)
    }

def get_price_bins(y):
    """Converts continuous prices into 3 statistical categories."""
    return pd.qcut(y, q=3, labels=["Low", "Mid", "High"])

def save_artifact(obj, filename):
    """Saves models or objects to the models/ folder."""
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(obj, f"models/{filename}.pkl")