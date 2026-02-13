import os
import joblib
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.models_complete import RegressionTrainer, ClassificationTrainer, ClusteringAnalysis

def save_model(model, name):
    """Helper to save model"""
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', f'{name}.pkl')
    joblib.dump(model, filepath)
    print(f"💾 Saved: {filepath}")

def run_training():
    """
    Step 3: Model Training
    - Load train/test datasets
    - Train Regression Model
    - Train Classification Model
    - Perform Clustering
    - Save trained models
    """
    print("\n" + "🚀"*30)
    print("STEP 3: MODEL TRAINING")
    print("🚀"*30)

    # 1. Load Data
    data_dir = 'data/processed'
    try:
        print("\n📥 Loading split data...")
        X_train = joblib.load(f'{data_dir}/X_train.joblib')
        X_test = joblib.load(f'{data_dir}/X_test.joblib')
        y_train = joblib.load(f'{data_dir}/y_train.joblib')
        y_test = joblib.load(f'{data_dir}/y_test.joblib')
    except FileNotFoundError:
        print("❌ Error: Processed data not found. Run Step 2 first.")
        return

    # ============================================================
    # TASK 1: REGRESSION
    # ============================================================
    print("\n" + "="*40)
    print("TASK 1: REGRESSION TRAINING")
    print("="*40)
    
    reg_trainer = RegressionTrainer()
    reg_trainer.train_all(X_train, y_train, X_test, y_test)
    save_model(reg_trainer.best_model, 'ebay_price_predictor')
    
    # 🆕 Feature Importance Plot
    if hasattr(X_train, 'columns'):
        reg_trainer.save_feature_importance(X_train.columns)

    # ============================================================
    # TASK 2: CLASSIFICATION
    # ============================================================
    print("\n" + "="*40)
    print("TASK 2: CLASSIFICATION TRAINING")
    print("="*40)
    
    # Generate ground truth for classification based on best regression model
    # (Or based on logic defined in trainer, but here we reuse the logic from original script)
    
    # We need predictions to form the classes (undervalued/fair/overvalued)
    y_train_pred = reg_trainer.best_model.predict(X_train)
    y_test_pred = reg_trainer.best_model.predict(X_test)
    
    clf_trainer = ClassificationTrainer()
    y_train_cat = clf_trainer.create_price_categories(y_train_pred, y_train)
    y_test_cat = clf_trainer.create_price_categories(y_test_pred, y_test)
    
    clf_trainer.train_all(X_train, y_train_cat, X_test, y_test_cat)
    save_model(clf_trainer.best_model, 'ebay_pricing_classifier')

    # ============================================================
    # TASK 3: CLUSTERING
    # ============================================================
    print("\n" + "="*40)
    print("TASK 3: CLUSTERING")
    print("="*40)
    
    # Ideally clustering is done on the whole dataset or training set
    # Using X (features) - we can combine X_train and X_test or just use X_train
    # For consistency with original script which likely used whole X, let's concatenate
    X_full = pd.concat([X_train, X_test])
    
    clustering = ClusteringAnalysis(n_clusters=4)
    clustering.fit(X_full.values)
    save_model(clustering, 'ebay_clustering')
    
    print("\n✅ All models trained and saved successfully.")

if __name__ == "__main__":
    run_training()
