import joblib
import os
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from utils.model_utils import save_artifact, get_price_bins

# Configuration du logging pour suivre l'avancée de la recherche
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_training_pipeline():
    """
    ÉTAPE 3 OPTIMISÉE : HYPERPARAMETER TUNING (GRIDSEARCH)
    ----------------------------------------------------
    Objectif : Ne pas se contenter des réglages par défaut, mais tester 
    mathématiquement les meilleures combinaisons d'arbres et de profondeur.
    """
    logging.info("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT OPTIMISÉ (GRIDSEARCH)")

    # 1. CHARGEMENT DES DONNÉES
    if not os.path.exists('data/processed/X_final.joblib'):
        logging.error("❌ Matrices X/y manquantes. Exécutez l'Étape 2 d'abord.")
        return

    X_train = joblib.load('data/processed/X_final.joblib')
    y_train = joblib.load('data/processed/y_final.joblib')

    # 2. DÉFINITION DE LA GRILLE DE PARAMÈTRES
    # On définit les options que le modèle va tester
    param_grid = {
        'n_estimators': [100, 200],      # Nombre d'arbres dans la forêt
        'max_depth': [10, 20, None],     # Profondeur des décisions
        'min_samples_split': [2, 5],     # Nombre min d'échantillons pour diviser un nœud
        'bootstrap': [True]              # Méthode d'échantillonnage
    }

    # 3. OPTIMISATION DE LA RÉGRESSION
    logging.info("🔍 Recherche des meilleurs paramètres pour la Régression...")
    rf_reg = RandomForestRegressor(random_state=42)
    
    # GridSearchCV divise les données en 3 (cv=3) pour valider chaque combinaison
    grid_reg = GridSearchCV(
        estimator=rf_reg, 
        param_grid=param_grid, 
        cv=3, 
        n_jobs=-1, 
        scoring='neg_mean_absolute_error',
        verbose=1
    )
    grid_reg.fit(X_train, y_train)
    
    logging.info(f"✨ Meilleurs paramètres Régression : {grid_reg.best_params_}")
    save_artifact(grid_reg.best_estimator_, "price_regressor")

    # 4. OPTIMISATION DE LA CLASSIFICATION
    logging.info("🔍 Recherche des meilleurs paramètres pour la Classification...")
    y_class = get_price_bins(y_train)
    rf_clf = RandomForestClassifier(random_state=42)
    
    grid_clf = GridSearchCV(
        estimator=rf_clf, 
        param_grid=param_grid, 
        cv=3, 
        n_jobs=-1, 
        scoring='accuracy',
        verbose=1
    )
    grid_clf.fit(X_train, y_class)
    
    logging.info(f"✨ Meilleurs paramètres Classification : {grid_clf.best_params_}")
    save_artifact(grid_clf.best_estimator_, "price_classifier")

    logging.info("✅ Étape 3 terminée : Modèles optimisés sauvegardés.")

if __name__ == "__main__":
    run_training_pipeline()