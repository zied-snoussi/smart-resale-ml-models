import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Smart Resale Predictor", page_icon="💰", layout="wide")

# Chargement des modèles avec cache pour la rapidité
@st.cache_resource
def load_ml_assets():
    return {
        'vectorizer': joblib.load('models/tfidf_svd.pkl'),
        'scaler': joblib.load('models/scaler.pkl'),
        'reg': joblib.load('models/price_regressor.pkl'),
        'clf': joblib.load('models/price_classifier.pkl')
    }

assets = load_ml_assets()

# --- INTERFACE ---
st.title("💰 Smart Resale : Optimisation de Prix")
st.markdown("Estimez instantanément la valeur de revente de vos produits Tech.")

with st.sidebar:
    st.header("⚙️ Paramètres de l'objet")
    input_msrp = st.number_input("Prix d'origine (MSRP €)", min_value=10.0, max_value=2000.0, value=450.0)
    is_bundle = st.toggle("L'annonce est-elle un lot/bundle ?", value=False)
    demand_score = st.slider("Indice de demande locale", 0.0, 1.0, 0.5)

# Zone de saisie principale
title = st.text_input("📝 Titre de l'annonce eBay", placeholder="ex: Apple iPhone 13 Pro 128GB Sierra Blue")

if title:
    # 1. Extraction Features
    num_features = {
        'msrp': input_msrp,
        'has_brand': 1 if any(b in title.upper() for b in ['APPLE', 'SAMSUNG', 'SONY', 'NINTENDO']) else 0,
        'is_bundle': 1 if is_bundle else 0,
        'title_len': len(title),
        'word_count': len(title.split()),
        'demand_score': demand_score
    }
    
    # 2. NLP
    tfidf = assets['vectorizer']['tfidf']
    svd = assets['vectorizer']['svd']
    text_svd = svd.transform(tfidf.transform([title]))
    text_features = {f'text_svd_{i}': val for i, val in enumerate(text_svd[0])}
    
    # 3. Alignement & Prédiction
    full_data = {**num_features, **text_features}
    X_input = pd.DataFrame([full_data])[assets['scaler'].feature_names_in_]
    X_scaled = assets['scaler'].transform(X_input)
    
    price = assets['reg'].predict(X_scaled)[0]
    category = assets['clf'].predict(X_scaled)[0]
    probs = assets['clf'].predict_proba(X_scaled)[0]

    # --- AFFICHAGE DES RÉSULTATS ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prix de Revente Estimé", f"{price:.2f} €")
    
    with col2:
        st.info(f"Segment : **{category}**")
        
    with col3:
        confidence = np.max(probs) * 100
        st.progress(confidence/100, text=f"Confiance : {confidence:.1f}%")

    # Graphique de probabilités
    st.divider()
    st.subheader("📊 Analyse des segments de prix")
    chart_data = pd.DataFrame({
        'Catégorie': ['Low', 'Mid', 'High'],
        'Probabilité': probs
    })
    st.bar_chart(chart_data, x='Catégorie', y='Probabilité', color='#3498db')

else:
    st.info("Saisissez un titre ci-dessus pour lancer l'analyse.")