import joblib
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.models_complete import RecommendationSystem

def run_evaluation():
    """
    Step 4: Evaluation & Recommendations
    - Load trained models
    - Load test data
    - Generate performance report
    - Test Recommendation Engine
    """
    print("\n" + "🚀"*30)
    print("STEP 4: EVALUATION & RECOMMENDATION")
    print("🚀"*30)

    # 1. Load Data & Models
    data_dir = 'data/processed'
    model_dir = 'models'
    
    try:
        print("\n📥 Loading resources...")
        X_test = joblib.load(f'{data_dir}/X_test.joblib')
        y_test = joblib.load(f'{data_dir}/y_test.joblib')
        
        reg_model = joblib.load(f'{model_dir}/ebay_price_predictor.pkl')
        clf_model = joblib.load(f'{model_dir}/ebay_pricing_classifier.pkl')
        clustering = joblib.load(f'{model_dir}/ebay_clustering.pkl')
    except list(FileNotFoundError): # type: ignore
        print("❌ Error: Missing data or models. Run previous steps.")
        return

    # 2. Recommendation Engine Test
    print("\n" + "="*40)
    print("TESTING RECOMMENDATIONS")
    print("="*40)
    
    recommender = RecommendationSystem(reg_model)
    
    # Save recommender explicitly if needed (it was wrapping the reg model)
    joblib.dump(recommender, f'{model_dir}/ebay_recommender.pkl')
    print(f"💾 Saved recommender to {model_dir}/ebay_recommender.pkl")

    # Show 5 examples
    print("\n💡 Sample Recommendations:\n")
    for i in range(min(5, len(X_test))):
        X_sample = X_test.iloc[i:i+1]
        current_price = y_test.iloc[i]
        
        recommendations = recommender.suggest_optimal_price(X_sample, current_price)
        
        print(f"Product #{i+1}:")
        print(f"  Current Price: €{current_price:.2f}")
        for rec in recommendations:
            print(f"  → {rec['action']}: €{rec['suggested_price']:.2f}")
            print(f"     Reason: {rec['reason']}")
        print()

    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    run_evaluation()
