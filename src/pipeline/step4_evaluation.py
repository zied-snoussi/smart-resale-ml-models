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

# ==============================================================================
# ÉTAPE 4 : DIAGNOSTIC VISUEL ET VALIDATION FINALE
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_step4_evaluation():
    """
    ÉTAPE 4 : ANALYSE DE PERFORMANCE
    -------------------------------
    Objectif : Produire des diagnostics visuels pour comprendre les forces 
    et les faiblesses des modèles de régression et de classification.
    """
    logging.info("🚀 DÉMARRAGE DE L'ÉVALUATION VISUELLE...")

    # 1. CHARGEMENT DES DONNÉES ET DES MODÈLES
    # On récupère les données de test (non vues à l'entraînement) pour une évaluation honnête.
    try:
        X_test = joblib.load('data/processed/X_final.joblib')
        y_test = joblib.load('data/processed/y_final.joblib')
        reg_model = joblib.load('models/price_regressor.pkl')
        clf_model = joblib.load('models/price_classifier.pkl')
    except FileNotFoundError as e:
        logging.error(f"❌ Erreur : Artefacts manquants. Assurez-vous d'avoir fini l'étape 3. {e}")
        return

    # 2. GÉNÉRATION DES PRÉDICTIONS
    # Prédiction des prix (Régression) et des segments de prix (Classification)
    predictions = reg_model.predict(X_test)
    
    # Transformation de la vérité terrain en classes pour comparer avec le classifieur
    y_test_class = pd.qcut(y_test, q=3, labels=["Low", "Mid", "High"])
    clf_predictions = clf_model.predict(X_test)

    # 3. GÉNÉRATION DES RAPPORTS GRAPHIQUES (MOTEUR VISUEL)
    logging.info("📸 Génération des rapports graphiques dans static/plots/...")
    
    # Graphique 1 : Erreur de Prédiction (Scatter Plot)
    # Vérifie si le modèle prédit bien les prix élevés comme les petits prix.
    plot_prediction_error(y_test, predictions)
    
    
    # Graphique 2 : Matrice de Confusion
    # Analyse les erreurs de segmentation (ex: un produit de luxe classé en "Mid").
    plot_confusion_matrix(y_test_class, clf_predictions)
    
    
    # Graphique 3 : Distribution des Résidus
    # Identifie si le modèle a tendance à surévaluer ou sous-évaluer les objets.
    plot_error_distribution(y_test, predictions)
    
    # Graphique 4 : Importance des Caractéristiques (Explainable AI)
    # Révèle quels facteurs (MSRP, Marque, NLP) influencent le plus le prix final.
    feature_names = X_test.columns.tolist() if hasattr(X_test, 'columns') else [f"Var_{i}" for i in range(X_test.shape[1])]
    plot_feature_importance(reg_model, feature_names)
    

    logging.info("✅ Évaluation terminée. Les résultats sont disponibles dans 'static/plots/'.")

if __name__ == "__main__":
    run_step4_evaluation()