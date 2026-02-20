import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# ==============================================================================
# MODULE D'ÉVALUATION ET DE PERSISTENCE DES MODÈLES
# ==============================================================================

def evaluate_regression(model, X, y):
    """
    Calcule les métriques de performance pour les modèles de régression.
    
    Indicateurs clés :
    - MAE (Mean Absolute Error) : L'écart moyen en Euros par rapport au prix réel.
    - R² : La capacité du modèle à expliquer la variance des prix (proche de 1.0).
    """
    preds = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, preds),
        "R2": r2_score(y, preds)
    }

def evaluate_classification(model, X, y):
    """
    Calcule les métriques de performance pour les modèles de classification.
    
    Indicateur clé :
    - Accuracy : Le pourcentage de prédictions correctes sur les segments de prix.
    """
    preds = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, preds)
    }

def get_price_bins(y):
    """
    Transforme les prix continus en 3 catégories statistiques (Tiers).
    Utilise la discrétisation par quantiles (qcut) pour garantir des classes équilibrées :
    - Low : 33% des produits les moins chers.
    - Mid : 33% des produits de milieu de gamme.
    - High : 33% des produits les plus chers.
    """
    return pd.qcut(y, q=3, labels=["Low", "Mid", "High"])

def save_artifact(obj, filename):
    """
    Sécurise et sauvegarde les modèles ou objets (scalers, vecteurs) sur le disque.
    Utilise joblib, plus performant que pickle pour les gros tableaux de données.
    
    Args:
        obj: L'objet Python à sauvegarder (modèle, dictionnaire, etc.).
        filename (str): Le nom du fichier sans extension.
    """
    # Garantie que le répertoire de destination existe pour éviter les erreurs d'écriture
    os.makedirs('models', exist_ok=True)
    
    save_path = f"models/{filename}.pkl"
    joblib.dump(obj, save_path)
    print(f"💾 Artefact sauvegardé avec succès : {save_path}")