import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# ==============================================================================
# CONFIGURATION GRAPHIQUE & ESTHÉTIQUE
# ==============================================================================

# Utilisation d'un style propre et moderne (ggplot) pour les présentations
plt.style.use('ggplot')

def plot_prediction_error(y_true, y_pred, save_path='static/plots'):
    """
    Génère un graphique de dispersion (Scatter Plot) comparant les valeurs réelles 
    et les prédictions.
    
    Utilité : Visualiser visuellement la variance et identifier si le modèle 
    décroche sur les hautes valeurs (prix élevés).
    """
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10, 7))
    
    # Nuage de points avec alpha pour gérer la superposition (densité)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4, color='#3498db')
    
    # Ligne de référence à 45° : représente la prédiction idéale (y = x)
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], color='#e74c3c', linestyle='--', lw=2, label='Prédiction Parfaite')
    
    plt.title('Précision de la Régression : Réel vs Prédit', fontsize=15)
    plt.xlabel('Prix Réel (€)', fontsize=12)
    plt.ylabel('Prix Prédit (€)', fontsize=12)
    plt.legend()
    
    output = os.path.join(save_path, 'prediction_error.png')
    plt.savefig(output, dpi=300) # Résolution 300 DPI pour publication
    plt.close()
    print(f"📈 Graphique d'erreur de prédiction sauvegardé : {output}")

def plot_confusion_matrix(y_true, y_pred, labels=["Low", "Mid", "High"], save_path='static/plots'):
    """
    Génère une matrice de confusion normalisée sous forme de Heatmap.
    
    Utilité : Identifier quelles catégories de prix sont confondues par le modèle 
    (ex: un produit 'High' classé en 'Mid').
    """
    # Calcul de la matrice et normalisation par ligne (pourcentages de rappel)
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
    
    print(f"📊 Matrice de confusion sauvegardée : {output}")
    # Export du rapport textuel complet (F1-score, Recall, Precision)
    print("\n--- RAPPORT DE CLASSIFICATION DÉTAILLÉ ---\n", classification_report(y_true, y_pred, target_names=labels))

def plot_error_distribution(y_true, y_pred, save_path='static/plots'):
    """
    Analyse de la distribution des résidus (erreurs de prédiction).
    
    Utilité : Vérifier l'homoscédasticité et l'absence de biais systématique.
    Une distribution centrée sur 0 et symétrique indique un modèle sain.
    """
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    
    # Histogramme combiné à une estimation de la densité par noyau (KDE)
    sns.histplot(errors, kde=True, color='#9b59b6', bins=50)
    # Ligne d'erreur nulle
    plt.axvline(x=0, color='red', linestyle='--', lw=2)
    
    plt.title('Distribution de l\'Erreur (Résidus)', fontsize=15)
    plt.xlabel('Erreur (€) - [Négatif = Sous-estimé | Positif = Surestimé]', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    
    output = os.path.join(save_path, 'error_distribution.png')
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"📊 Distribution des résidus sauvegardée : {output}")

def plot_feature_importance(model, feature_names, save_path='static/plots'):
    """
    Visualisation de l'importance des variables (Feature Importance).
    
    Utilité : Expliquer le modèle ('Explainable AI'). Permet de savoir si 
    le modèle se base sur le MSRP, la marque ou l'état pour décider du prix.
    """
    # Extraction des poids d'importance spécifiques aux modèles basés sur les arbres
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Tri des 15 variables les plus impactantes
        indices = np.argsort(importances)[-15:] 
        
        plt.figure(figsize=(10, 8))
        plt.title('Top 15 des Variables Influentes', fontsize=15)
        plt.barh(range(len(indices)), importances[indices], color='#2ecc71', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance Relative', fontsize=12)
        
        output = os.path.join(save_path, 'feature_importance.png')
        plt.savefig(output, dpi=300)
        plt.close()
        print(f"📊 Importance des variables sauvegardée : {output}")