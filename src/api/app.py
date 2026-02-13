import sys
import os

# Add src to python path to allow importing from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re

# Import preprocessing utils
try:
    from src.utils.preprocessing import extract_features_ebay
except ImportError:
    # Fallback if running from src/api
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.utils.preprocessing import extract_features_ebay

app = Flask(__name__)
CORS(app)

print("Loading models and vectorizers...")
try:
    models = {
        'price_predictor': joblib.load('models/ebay_price_predictor.pkl'),
        'pricing_classifier': joblib.load('models/ebay_pricing_classifier.pkl'),
        'recommender': joblib.load('models/ebay_recommender.pkl'),
        # 'clustering': joblib.load('models/ebay_clustering.pkl') # Optional
    }
    tfidf_vectorizer = joblib.load('models/tfidf.pkl')
    svd_transformer = joblib.load('models/svd.pkl')
    print("✓ Models & Vectorizers loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # Create dummies for dev/testing if files missing
    models = {} 

def prepare_features(data):
    """
    Full feature extraction pipeline for inference.
    Replicates steps from src/utils/preprocessing.py and src/pipeline/step2_features.py
    """
    # 1. Create a 1-row DataFrame mimicking the raw clean data
    # Note: 'average_rating' and 'num_reviews' are expected by extract_features_ebay
    input_row = {
        'Title': data.get('title', ''),
        'price_cleaned': float(data.get('current_price', 0)),
        'average_rating': float(data.get('average_rating', 0)),
        'num_reviews': float(data.get('num_reviews', 0)),
        # Preprocessing helper needs these originally, but extract_features_ebay uses the clean names
        # We provide clean names directly
        'One Star': 0, 
        'Number Of Ratings': 1 # Avoid div/0
    }
    
    df_input = pd.DataFrame([input_row])

    # 2. Apply rules-based feature extraction (Brand, Regex, etc.)
    features_df = extract_features_ebay(df_input)
    
    # 3. Add Screen Size & Memory (Custom logic from app.py)
    # Prefer explicit input, fallback to regex extraction if input is 0 or missing
    if data.get('screen_size', 0) > 0:
        features_df['screen_size'] = float(data.get('screen_size'))
        
    if data.get('memory_gb', 0) > 0:
        features_df['memory_gb'] = float(data.get('memory_gb'))

    # 4. Semantic Features (TF-IDF + SVD)
    if tfidf_vectorizer and svd_transformer:
        titles = df_input['Title'].fillna("").astype(str)
        tfidf_matrix = tfidf_vectorizer.transform(titles)
        svd_matrix = svd_transformer.transform(tfidf_matrix)
        
        # Create column names matching training
        for i in range(svd_matrix.shape[1]):
            features_df[f'text_svd_{i}'] = svd_matrix[0, i]
            
    # 5. Add Missing Enriched Features (Depreciation, Amazon Demand)
    # These require the full catalog lookup, which we skip for API latency.
    # We fill with NaN (XGBoost handles this) or safe defaults.
    if 'depreciation_pct' not in features_df.columns:
        features_df['depreciation_pct'] = np.nan 
    if 'amazon_demand' not in features_df.columns:
        features_df['amazon_demand'] = np.nan

    # 6. Ensure column order/existence matches Model Expectation
    # The models objects (XGBoost/RF) are sensitive to column order if passed as array, 
    # but slightly more robust as DataFrame if names match.
    # However, Scikit-Learn transformers might strip names. 
    # Ideally we should re-index against the training columns.
    # For now, we rely on the classifier matching names.
    
    # The training removed 'current_price' and 'log_price' from X.
    # We must drop them if they exist in features
    cols_to_drop = ['current_price', 'log_price']
    X_inference = features_df.drop([c for c in cols_to_drop if c in features_df.columns], axis=1)
    
    return X_inference

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

import traceback

# ... imports ...

# ...

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict resale price and provide recommendations
    """
    try:
        data = request.json
        print(f"Received data: {data}")
        
        # Prepare features using the full pipeline
        X = prepare_features(data)
        print(f"Prepared features shape: {X.shape}")
        print(f"Feature columns: {X.columns.tolist()}")
        
        current_price = data.get('current_price', 0)
        
        # 1. Predict price
        if 'price_predictor' not in models:
             raise ValueError("Price predictor model not loaded")

        # Align features with model expectations
        predictor = models['price_predictor']
        if hasattr(predictor, 'feature_names_in_'):
            expected_cols = predictor.feature_names_in_
            # Add missing columns with 0
            missing_cols = set(expected_cols) - set(X.columns)
            if missing_cols:
                print(f"⚠️ Warning: Missing features filled with 0: {missing_cols}")
                for c in missing_cols:
                    X[c] = 0
            
            # Reorder and filter columns
            X = X[list(expected_cols)]
            
        # Fill NaNs for models that don't support them (like SVC)
        # Using 0 as safe default for missing numeric features
        X = X.fillna(0)
            
        predicted_price = predictor.predict(X)[0]
        
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
        print("❌ Error during prediction:")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 400

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Resale Price Predictor API")
    print("="*60)
    print("\nAPI running at: http://localhost:5001")
    print("\nEndpoints:")
    print("  GET  /         - API info")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Predict price")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)