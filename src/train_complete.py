import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

from load_data import load_ebay_data, load_amazon_products
from preprocessing import (preprocess_ebay_data, preprocess_amazon_data,
                          extract_features_ebay, extract_features_amazon,
                          prepare_train_test_split)
from models_complete import (RegressionTrainer, ClassificationTrainer,
                             RecommendationSystem, ClusteringAnalysis)

def save_model(model, name):
    """Save trained model"""
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', f'{name}.pkl')
    joblib.dump(model, filepath)
    print(f"💾 Saved: {filepath}")

def train_ebay_pipeline():
    """Complete ML pipeline for eBay data"""
    print("\n" + "🚀"*30)
    print("RESALE PRICE PREDICTOR - EBAY DATASET")
    print("🚀"*30)
    
    # 1. LOAD & PREPROCESS DATA
    print("\n📥 Step 1: Loading and preprocessing data...")
    df = load_ebay_data()
    df_clean = preprocess_ebay_data(df)
    features = extract_features_ebay(df_clean)
    
    # Prepare data for models (exclude price from X)
    X = features.drop(['current_price', 'log_price'], axis=1)
    y = features['current_price']
    
    # Split data
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y, test_size=0.2)
    
    # ============================================================
    # OBJECTIF 1: RÉGRESSION - Prédire le prix de revente
    # ============================================================
    print("\n" + "="*60)
    print("OBJECTIF 1: PRÉDICTION DU PRIX DE REVENTE")
    print("="*60)
    
    reg_trainer = RegressionTrainer()
    reg_results = reg_trainer.train_all(X_train, y_train, X_test, y_test)
    
    # Save best regression model
    save_model(reg_trainer.best_model, 'ebay_price_predictor')
    
    # ============================================================
    # OBJECTIF 2: CLASSIFICATION - Évaluer le pricing
    # ============================================================
    print("\n" + "="*60)
    print("OBJECTIF 2: CLASSIFICATION DU PRICING")
    print("="*60)
    
    # Get predictions from best regression model
    y_train_pred = reg_trainer.best_model.predict(X_train)
    y_test_pred = reg_trainer.best_model.predict(X_test)
    
    # Create categories
    clf_trainer = ClassificationTrainer()
    y_train_cat = clf_trainer.create_price_categories(y_train_pred, y_train)
    y_test_cat = clf_trainer.create_price_categories(y_test_pred, y_test)
    
    print(f"\n📊 Distribution des catégories:")
    print(f"   Sous-évalué (0): {(y_test_cat == 0).sum()}")
    print(f"   Juste prix (1): {(y_test_cat == 1).sum()}")
    print(f"   Surévalué (2): {(y_test_cat == 2).sum()}\n")
    
    clf_results = clf_trainer.train_all(X_train, y_train_cat, X_test, y_test_cat)
    
    # Save best classification model
    save_model(clf_trainer.best_model, 'ebay_pricing_classifier')
    
    # ============================================================
    # OBJECTIF 3: RECOMMANDATION - Suggérer le meilleur prix
    # ============================================================
    print("\n" + "="*60)
    print("OBJECTIF 3: SYSTÈME DE RECOMMANDATION")
    print("="*60)
    
    recommender = RecommendationSystem(reg_trainer.best_model)
    
    # Test recommendations on 5 random products
    print("\n💡 Exemples de recommandations:\n")
    for i in range(min(5, len(X_test))):
        X_sample = X_test.iloc[i:i+1]
        current_price = y_test.iloc[i]
        
        recommendations = recommender.suggest_optimal_price(X_sample, current_price)
        
        print(f"Produit #{i+1}:")
        print(f"  Prix actuel: €{current_price:.2f}")
        for rec in recommendations:
            print(f"  → {rec['action']}: €{rec['suggested_price']:.2f}")
            print(f"     Raison: {rec['reason']}")
            print(f"     Impact: {rec['impact']}")
        print()
    
    # Save recommender
    save_model(recommender, 'ebay_recommender')
    
    # ============================================================
    # OBJECTIF 4: CLUSTERING - Segmenter les produits
    # ============================================================
    print("\n" + "="*60)
    print("OBJECTIF 4: CLUSTERING & SEGMENTATION")
    print("="*60)
    
    # Use all features for clustering
    clustering = ClusteringAnalysis(n_clusters=4)
    cluster_labels = clustering.fit(X.values)
    
    # Analyze clusters
    print("\n📦 Profils des clusters:\n")
    df_clean['cluster'] = cluster_labels
    
    for cluster_id in range(4):
        cluster_data = df_clean[df_clean['cluster'] == cluster_id]
        avg_price = cluster_data['price_cleaned'].mean()
        avg_rating = cluster_data['average_rating'].mean()
        
        print(f"Cluster {cluster_id + 1}:")
        print(f"  Taille: {len(cluster_data)} produits")
        print(f"  Prix moyen: €{avg_price:.2f}")
        print(f"  Note moyenne: {avg_rating:.2f}/5")
        
        # Identify cluster characteristics
        if avg_price > df_clean['price_cleaned'].median() * 1.5:
            print(f"  Type: 📱 Produits premium à valeur stable")
        elif avg_rating >= 4.5:
            print(f"  Type: ⭐ Produits haute qualité")
        elif avg_price < df_clean['price_cleaned'].median() * 0.5:
            print(f"  Type: 💰 Produits à forte dépréciation")
        else:
            print(f"  Type: 📦 Produits standard")
        print()
    
    # Save clustering model
    save_model(clustering, 'ebay_clustering')
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "✅"*30)
    print("RÉSUMÉ DU TRAINING")
    print("✅"*30)
    
    print(f"\n📊 Meilleurs modèles:")
    print(f"   Régression: {min(reg_results, key=lambda x: reg_results[x]['rmse'])}")
    print(f"   Classification: {max(clf_results, key=lambda x: clf_results[x]['accuracy'])}")
    
    print(f"\n💾 Modèles sauvegardés dans /models:")
    for filename in os.listdir('models'):
        print(f"   - {filename}")
    
    print(f"\n📈 Statistiques:")
    print(f"   Dataset: {len(df_clean):,} produits")
    print(f"   Features: {len(X.columns)}")
    print(f"   Train/Test split: 80/20")
    
    return {
        'regression': reg_results,
        'classification': clf_results,
        'recommender': recommender,
        'clustering': clustering
    }

def train_amazon_pipeline():
    """Complete ML pipeline for Amazon data"""
    print("\n" + "🚀"*30)
    print("RESALE PRICE PREDICTOR - AMAZON DATASET")
    print("🚀"*30)
    
    # Load and preprocess (sample for speed)
    print("\n📥 Loading Amazon data (sampling 50k for speed)...")
    df = load_amazon_products()
    df_sample = df.sample(n=50000, random_state=42)
    
    df_clean = preprocess_amazon_data(df_sample)
    features = extract_features_amazon(df_clean)
    
    # Prepare data
    X = features.drop(['current_price', 'log_price'], axis=1)
    y = features['current_price']
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)
    
    # Train regression model
    reg_trainer = RegressionTrainer()
    reg_results = reg_trainer.train_all(X_train, y_train, X_test, y_test)
    
    save_model(reg_trainer.best_model, 'amazon_price_predictor')
    
    print(f"\n✅ Amazon model training complete!")
    
    return reg_results

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("RESALE PRICE PREDICTOR - ML TRAINING PIPELINE")
    print("Projet: Pricing Dynamique & Dépréciation Intelligente")
    print("="*60)
    
    # Train on eBay data (complete pipeline)
    ebay_results = train_ebay_pipeline()
    
    # Optional: Train on Amazon data
    print("\n" + "❓"*30)
    response = input("Train sur Amazon dataset aussi? (y/n): ")
    if response.lower() == 'y':
        amazon_results = train_amazon_pipeline()
    
    print("\n" + "🎉"*30)
    print("TRAINING TERMINÉ AVEC SUCCÈS!")
    print("🎉"*30)
    print("\nProchaines étapes:")
    print("  1. Tester les modèles avec predict.py")
    print("  2. Créer une API Flask/FastAPI")
    print("  3. Intégrer dans l'interface web")

if __name__ == "__main__":
    main()