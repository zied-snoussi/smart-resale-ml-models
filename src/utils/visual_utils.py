import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Configuration du style pour des images "Clean"
plt.style.use('ggplot')

def plot_prediction_error(y_true, y_pred, save_path='static/plots'):
    """Graphique de dispersion : Idéal pour voir la précision globale."""
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4, color='#3498db')
    
    # Ligne de perfection
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], color='#e74c3c', linestyle='--', lw=2, label='Prédiction Parfaite')
    
    plt.title('Précision de la Régression : Réel vs Prédit', fontsize=15)
    plt.xlabel('Prix Réel (€)', fontsize=12)
    plt.ylabel('Prix Prédit (€)', fontsize=12)
    plt.legend()
    
    output = os.path.join(save_path, 'prediction_error.png')
    plt.savefig(output, dpi=300) # Haute résolution
    plt.close()
    print(f"📈 Image sauvegardée : {output}")

def plot_confusion_matrix(y_true, y_pred, labels=["Low", "Mid", "High"], save_path='static/plots'):
    """Matrice de confusion avec pourcentages : Idéal pour voir les erreurs par classe."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.1%', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    
    plt.title('Matrice de Confusion (Précision par Tiers)', fontsize=15)
    plt.xlabel('Prédictions du Modèle', fontsize=12)
    plt.ylabel('Réalité (Vérité)', fontsize=12)
    
    output = os.path.join(save_path, 'confusion_matrix_detailed.png')
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"📊 Image sauvegardée : {output}")
    print("\n--- RAPPORT DÉTAILLÉ ---\n", classification_report(y_true, y_pred, target_names=labels))

def plot_error_distribution(y_true, y_pred, save_path='static/plots'):
    """Distribution des résidus : Idéal pour voir si le modèle est biaisé."""
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, color='#9b59b6', bins=50)
    plt.axvline(x=0, color='red', linestyle='--', lw=2)
    
    plt.title('Distribution de l\'Erreur (Résidus)', fontsize=15)
    plt.xlabel('Erreur (€) - [Négatif = Sous-estimé | Positif = Surestimé]', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    
    output = os.path.join(save_path, 'error_distribution.png')
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"📊 Image sauvegardée : {output}")

def plot_feature_importance(model, feature_names, save_path='static/plots'):
    """Importance des variables : Idéal pour comprendre 'comment' le modèle décide."""
    # Fonctionne pour les modèles basés sur les arbres (RandomForest, XGBoost, etc.)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:] # Top 15 variables
        
        plt.figure(figsize=(10, 8))
        plt.title('Top 15 des Variables Influentes', fontsize=15)
        plt.barh(range(len(indices)), importances[indices], color='#2ecc71', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance Relative', fontsize=12)
        
        output = os.path.join(save_path, 'feature_importance.png')
        plt.savefig(output, dpi=300)
        plt.close()
        print(f"📊 Image sauvegardée : {output}")