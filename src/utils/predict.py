import joblib
import pandas as pd
import numpy as np

def load_models():
    """Load all trained models"""
    models = {
        'price_predictor': joblib.load('models/ebay_price_predictor.pkl'),
        'pricing_classifier': joblib.load('models/ebay_pricing_classifier.pkl'),
        'recommender': joblib.load('models/ebay_recommender.pkl'),
        'clustering': joblib.load('models/ebay_clustering.pkl')
    }
    return models

def prepare_features(product_data):
    """
    Prepare features in the exact same format as training
    
    Training features (from preprocessing.py):
    ['average_rating', 'num_reviews', 'log_reviews', 'has_reviews', 
     'title_length', 'word_count', 'is_new', 'is_used', 'is_refurbished', 
     'has_brand', 'screen_size', 'memory_gb']
    """
    num_reviews = product_data.get('num_reviews', 0)
    
    features = {
        'average_rating': product_data.get('average_rating', 0),
        'num_reviews': num_reviews,
        'log_reviews': np.log1p(num_reviews),  # CRITICAL: This was missing
        'has_reviews': 1 if num_reviews > 0 else 0,
        'title_length': product_data.get('title_length', 50),
        'word_count': product_data.get('word_count', 5),
        'is_new': product_data.get('is_new', 0),
        'is_used': product_data.get('is_used', 1),
        'is_refurbished': product_data.get('is_refurbished', 0),
        'has_brand': product_data.get('has_brand', 0),
        'screen_size': product_data.get('screen_size', 0),
        'memory_gb': product_data.get('memory_gb', 0)
    }
    
    return features

def predict_resale_price(product_features):
    """
    Prédire le prix de revente d'un produit
    
    Args:
        product_features: dict avec les caractéristiques du produit
        
    Returns:
        dict avec prédictions et recommandations
    """
    models = load_models()
    
    # Prepare features in correct format
    features = prepare_features(product_features)
    X = pd.DataFrame([features])
    
    # Get the current price (not used as a feature, only for recommendations)
    current_price = product_features.get('current_price', 0)
    
    # 1. Prédiction du prix
    predicted_price = models['price_predictor'].predict(X)[0]
    
    # 2. Classification du pricing
    pricing_class = models['pricing_classifier'].predict(X)[0]
    pricing_labels = ['Sous-évalué', 'Juste prix', 'Surévalué']
    
    # 3. Recommandation
    recommendations = models['recommender'].suggest_optimal_price(
        X, 
        current_price if current_price > 0 else predicted_price
    )
    
    # 4. Cluster du produit
    cluster = models['clustering'].model.predict(X)[0]
    
    return {
        'predicted_price': round(predicted_price, 2),
        'pricing_category': pricing_labels[pricing_class],
        'recommendations': recommendations,
        'product_cluster': int(cluster) + 1,
        'features_used': list(features.keys())
    }

def example_prediction():
    """Exemple d'utilisation"""
    
    # Exemple: iPhone d'occasion
    product = {
        'average_rating': 4.5,
        'num_reviews': 150,
        'title_length': 45,
        'word_count': 8,
        'is_new': 0,
        'is_used': 1,
        'is_refurbished': 0,
        'has_brand': 1,
        'screen_size': 6.1,
        'memory_gb': 128,
        'current_price': 550  # Prix demandé par le vendeur
    }
    
    print("\n" + "="*60)
    print("PRÉDICTION DE PRIX DE REVENTE")
    print("="*60)
    
    print("\n📱 Produit:")
    print(f"  État: Utilisé")
    print(f"  Note: {product['average_rating']}/5 ({product['num_reviews']} avis)")
    print(f"  Mémoire: {product['memory_gb']}GB")
    print(f"  Prix demandé: €{product['current_price']}")
    
    result = predict_resale_price(product)
    
    print(f"\n🎯 Résultats:")
    print(f"  Prix estimé: €{result['predicted_price']}")
    print(f"  Catégorie: {result['pricing_category']}")
    print(f"  Segment: Cluster {result['product_cluster']}")
    
    print(f"\n💡 Recommandations:")
    for rec in result['recommendations']:
        print(f"  → {rec['action']}")
        print(f"     Prix suggéré: €{rec['suggested_price']:.2f}")
        print(f"     {rec['reason']}")
        print(f"     Impact: {rec['impact']}")
    
    print(f"\n🔧 Features utilisées:")
    print(f"  {', '.join(result['features_used'])}")

if __name__ == "__main__":
    example_prediction()