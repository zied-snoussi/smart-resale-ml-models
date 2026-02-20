import joblib
import pandas as pd
import logging
import numpy as np
from utils.visual_utils import (
    plot_prediction_error, 
    plot_confusion_matrix, 
    plot_error_distribution,
    plot_feature_importance
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_step4_evaluation():
    logging.info("🚀 DÉMARRAGE DE L'ÉVALUATION VISUELLE...")

    # 1. CHARGEMENT
    X_test = joblib.load('data/processed/X_final.joblib')
    y_test = joblib.load('data/processed/y_final.joblib')
    reg_model = joblib.load('models/price_regressor.pkl')
    clf_model = joblib.load('models/price_classifier.pkl')

    # 2. CALCULS
    predictions = reg_model.predict(X_test)
    y_test_class = pd.qcut(y_test, q=3, labels=["Low", "Mid", "High"])
    clf_predictions = clf_model.predict(X_test)

    # 3. GÉNÉRATION DES IMAGES
    logging.info("📸 Génération des rapports graphiques...")
    
    # Image 1: Dispersion (Est-ce que les points suivent la ligne rouge ?)
    plot_prediction_error(y_test, predictions)
    
    # Image 2: Matrice de Confusion (Quelles catégories on mélange ?)
    plot_confusion_matrix(y_test_class, clf_predictions)
    
    # Image 3: Distribution d'erreur (Est-on optimiste ou pessimiste ?)
    plot_error_distribution(y_test, predictions)
    
    # Image 4: Importance des colonnes (C'est le "cerveau" du modèle)
    feature_names = X_test.columns.tolist() if hasattr(X_test, 'columns') else [f"Feature {i}" for i in range(X_test.shape[1])]
    plot_feature_importance(reg_model, feature_names)

    logging.info("✅ Évaluation terminée. Consultez le dossier 'static/plots/'")

if __name__ == "__main__":
    run_step4_evaluation()