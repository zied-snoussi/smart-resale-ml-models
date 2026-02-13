import sys
import os

# Add src to python path to allow importing from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Updated import paths
# Note: Models are loaded assuming execution from root or with proper path handling

print("Loading models...")
models = {
    'price_predictor': joblib.load('models/ebay_price_predictor.pkl'),
    'pricing_classifier': joblib.load('models/ebay_pricing_classifier.pkl'),
    'recommender': joblib.load('models/ebay_recommender.pkl'),
    'clustering': joblib.load('models/ebay_clustering.pkl')
}
print("✓ Models loaded successfully!")

def prepare_features(data):
    """Prepare features matching training format"""
    num_reviews = data.get('num_reviews', 0)
    
    features = {
        'average_rating': data.get('average_rating', 0),
        'num_reviews': num_reviews,
        'log_reviews': np.log1p(num_reviews),  # CRITICAL
        'has_reviews': 1 if num_reviews > 0 else 0,
        'title_length': data.get('title_length', 50),
        'word_count': data.get('word_count', 5),
        'is_new': data.get('is_new', 0),
        'is_used': data.get('is_used', 1),
        'is_refurbished': data.get('is_refurbished', 0),
        'has_brand': data.get('has_brand', 0),
        'screen_size': data.get('screen_size', 0),
        'memory_gb': data.get('memory_gb', 0)
    }
    
    return features

@app.route('/')
def home():
    return jsonify({
        "message": "Resale Price Predictor API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predict resale price",
            "/health": "GET - Health check"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "models_loaded": len(models)})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict resale price and provide recommendations
    
    Example request:
    {
        "average_rating": 4.5,
        "num_reviews": 150,
        "title_length": 45,
        "word_count": 8,
        "is_new": 0,
        "is_used": 1,
        "is_refurbished": 0,
        "has_brand": 1,
        "screen_size": 6.1,
        "memory_gb": 128,
        "current_price": 550
    }
    """
    try:
        data = request.json
        
        # Prepare features in correct format
        features = prepare_features(data)
        X = pd.DataFrame([features])
        
        current_price = data.get('current_price', 0)
        
        # 1. Predict price
        predicted_price = models['price_predictor'].predict(X)[0]
        
        # 2. Classify pricing
        pricing_class = models['pricing_classifier'].predict(X)[0]
        pricing_labels = ['Sous-évalué', 'Juste prix', 'Surévalué']
        
        # 3. Get recommendations
        recommendations = models['recommender'].suggest_optimal_price(
            X, 
            current_price if current_price > 0 else predicted_price
        )
        
        # 4. Get cluster
        cluster = models['clustering'].model.predict(X)[0]
        
        return jsonify({
            "success": True,
            "predicted_price": round(float(predicted_price), 2),
            "pricing_category": pricing_labels[int(pricing_class)],
            "cluster": int(cluster) + 1,
            "recommendations": recommendations,
            "input": data
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Resale Price Predictor API")
    print("="*60)
    print("\nAPI running at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /         - API info")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Predict price")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)