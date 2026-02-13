import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

def visualize_model_comparison():
    """Visualize model performance comparison"""
    # This would use your actual results
    models = ['Linear Reg', 'Random Forest', 'Gradient Boost', 'SVR']
    rmse = [150.5, 95.3, 89.7, 112.4]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, rmse, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    plt.title('Comparaison des Modèles de Régression', fontsize=16, fontweight='bold')
    plt.xlabel('Modèle')
    plt.ylabel('RMSE (€)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=300)
    print("📊 Saved: models/model_comparison.png")

if __name__ == "__main__":
    visualize_model_comparison()