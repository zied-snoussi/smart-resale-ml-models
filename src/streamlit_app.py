import streamlit as st
import requests
import json

# Configuration
API_URL = "http://localhost:5000/predict"

st.set_page_config(page_title="Smart Resale Price Predictor", page_icon="💰", layout="wide")

st.title("🤖 Smart Resale Price Predictor")
st.markdown("""
This tool uses advanced Machine Learning (Random Forest & XGBoost) to predict the optimal resale price for electronics.
It benchmarks against a 1.4 million product catalog from Amazon.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Product Details")
    title = st.text_input("Product Title", placeholder="e.g. Apple iPad Air 2 64GB WiFi")
    
    c1, c2 = st.columns(2)
    current_price = c1.number_input("Current Listing Price (€)", min_value=0.0, value=100.0)
    num_reviews = c2.number_input("Number of Reviews", min_value=0, value=10)
    
    rating = st.slider("Average Rating", 1.0, 5.0, 4.5)
    
    condition = st.selectbox("Condition", ["Used", "New", "Refurbished"])
    
    st.subheader("Technical Specs (Optional)")
    screen_size = st.number_input("Screen Size (inches)", value=0.0)
    memory = st.number_input("Memory (GB)", value=0)

with col2:
    st.header("Prediction & Advice")
    
    if st.button("🚀 Analyze Price", type="primary"):
        # Construct payload logic matches api/app.py expectations
        is_new = 1 if condition == "New" else 0
        is_used = 1 if condition == "Used" else 0
        is_refurbished = 1 if condition == "Refurbished" else 0
        
        has_brand = 1 if any(b in title.lower() for b in ['apple', 'samsung', 'sony', 'dell', 'hp']) else 0
        
        payload = {
            "title_length": len(title),
            "word_count": len(title.split()),
            "is_new": is_new,
            "is_used": is_used,
            "is_refurbished": is_refurbished,
            "has_brand": has_brand,
            "average_rating": rating,
            "num_reviews": num_reviews,
            "screen_size": screen_size,
            "memory_gb": memory,
            "current_price": current_price
        }
        
        try:
            with st.spinner("Consulting AI Oracle..."):
                response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                pred_price = data.get('predicted_price', 0)
                category = data.get('pricing_category', 'Unknown')
                recs = data.get('recommendations', [])
                
                # Big Number Display
                st.metric(label="Predicted Market Value", value=f"€{pred_price:.2f}", delta=f"€{pred_price - current_price:.2f}")
                
                # Badge
                if category == "Juste prix":
                    st.success(f"✅ Verdict: {category}")
                elif category == "Sous-évalué":
                    st.warning(f"📉 Verdict: {category} (You can charge more!)")
                else:
                    st.error(f"📈 Verdict: {category} (Too expensive)")
                    
                # Recommendations
                st.subheader("💡 AI Recommendations")
                for rec in recs:
                    st.info(f"**{rec['action']}**: {rec['reason']}")
                    
            else:
                st.error(f"Error connecting to API: {response.status_code}")
                
        except Exception as e:
            st.error(f"Connection Failed. Is the Flask API running? ({e})")

st.markdown("---")
st.markdown("*Powered by Scikit-Learn, XGBoost, and Flask*")