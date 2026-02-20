import pandas as pd
import joblib
import os
import logging
from utils.feature_utils import (
    extract_numerical_features, 
    process_text_vectors, 
    scale_features
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_step2_features():
    """
    STEP 2: FEATURE ENGINEERING
    - Objective: Convert text to SVD vectors and scale numerical data.
    """
    logging.info("--- STARTING STEP 2: FEATURE ENGINEERING ---")

    # 1. LOAD PREPARED DATA
    input_path = 'data/processed/ebay_prep.pkl'
    if not os.path.exists(input_path):
        logging.error("Prep file not found! Run Step 1 first.")
        return
    
    df = pd.read_pickle(input_path)

    # 2. NUMERICAL FEATURE EXTRACTION
    # Creates features like 'is_bundle', 'title_length', 'brand_encoded'
    logging.info("Extracting numerical and categorical features...")
    df_features = extract_numerical_features(df)

    # 3. TEXT VECTORIZATION (The "Professional" NLP Step)
    # Uses TF-IDF + SVD to turn the 'Title' into 20 numeric columns
    logging.info("Generating Semantic Text Vectors (SVD)...")
    X_text, vectorizer_artifacts = process_text_vectors(df['Title'])
    
    # 4. COMBINE & SCALE
    # Merging numbers + text vectors and applying StandardScaler
    X_combined = pd.concat([df_features, X_text], axis=1)
    y = df['price_cleaned']
    
    logging.info(f"Scaling {X_combined.shape[1]} features...")
    X_scaled, scaler = scale_features(X_combined)

    # 5. SAVE ARTIFACTS
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Save the data for Training (Step 3)
    joblib.dump(X_scaled, 'data/processed/X_final.joblib')
    joblib.dump(y, 'data/processed/y_final.joblib')
    
    # Save the "Transformers" for use in the API/Streamlit later
    joblib.dump(vectorizer_artifacts, 'models/tfidf_svd.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    logging.info(f"✅ Step 2 Complete! Features ready: {X_scaled.shape[1]}")

if __name__ == "__main__":
    run_step2_features()