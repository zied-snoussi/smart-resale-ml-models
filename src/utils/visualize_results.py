import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os

# ==============================================================================
# MODULE DE BENCHMARKING ET VISUALISATION DES PERFORMANCES
# ==============================================================================

def visualize_model_comparison():
    """
    Génère un graphique à barres pour comparer les performances de différents modèles.
    
    Indicateur utilisé : RMSE (Root Mean Squared Error).
    Plus la valeur est basse, plus la précision du modèle de prix est élevée.
    """
    
    # Données simulées pour le benchmark (À remplacer par vos résultats réels)
    # ----------------------------------------------------------------------
    models = ['Linear Reg', 'Random Forest', 'Gradient Boost', 'SVR']
    rmse = [150.5, 95.3, 89.7, 112.4]
    
    # Initialisation de la figure avec une taille adaptée aux rapports
    plt.figure(figsize=(10, 6))
    
    # Création du barplot avec une palette de couleurs distinctes
    # Couleurs hexadécimales professionnelles (Flat Design)
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = plt.bar(models, rmse, color=colors)
    
    # Personnalisation esthétique
    plt.title('Comparaison de la Précision des Modèles (RMSE)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Algorithmes de Machine Learning', fontsize=12)
    plt.ylabel('Erreur Moyenne (RMSE en €)', fontsize=12)
    
    # Ajout d'une grille horizontale pour faciliter la lecture des valeurs
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Optimisation de l'espacement pour éviter les textes coupés
    plt.tight_layout()
    
    # Gestion sécurisée du répertoire de sauvegarde
    output_dir = 'models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, 'model_comparison.png')
    
    # Sauvegarde en haute résolution (300 DPI) pour une insertion propre dans un PDF/Web
    plt.savefig(save_path, dpi=300)
    plt.close() # Libère la mémoire vive
    
    print(f"📊 Graphique de comparaison sauvegardé : {save_path}")

# ==============================================================================
# POINT D'ENTRÉE DU SCRIPT
# ==============================================================================

if __name__ == "__main__":
    # Vérification du style Seaborn pour un rendu plus moderne
    sns.set_theme(style="whitegrid")
    visualize_model_comparison()