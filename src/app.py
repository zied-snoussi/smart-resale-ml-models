import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# 1. CONFIGURATION ET STYLE (Glassmorphism Dark Mode)
st.set_page_config(page_title="Smart Resale AI", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    /* Suppression du fond blanc des métriques pour un effet translucide */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
    }
    /* Harmonisation des textes de la sidebar */
    .stSlider label, .stSelectSlider label { color: #bdc3c7 !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. CHARGEMENT (Silencieux)
@st.cache_resource
def load_ml_assets():
    return {
        'vectorizer': joblib.load('models/tfidf_svd.pkl'),
        'scaler': joblib.load('models/scaler.pkl'),
        'reg': joblib.load('models/price_regressor.pkl'),
        'clf': joblib.load('models/price_classifier.pkl')
    }

assets = load_ml_assets()

# 3. SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    input_msrp = st.slider("💰 Prix Neuf Estimé (€)", 10, 2000, 450)
    demand_score = st.select_slider(
        "📈 Demande du Marché", 
        options=[0.1, 0.3, 0.5, 0.7, 0.9],
        value=0.5,
        format_func=lambda x: {0.1: "Bas", 0.5: "Normal", 0.9: "Critique"}.get(x, "Normal")
    )
    is_bundle = st.toggle("📦 Mode Bundle (Lot)", value=False)
    st.divider()
    st.info("💡 Un titre précis incluant la marque et le modèle améliore l'estimation.")

# 4. ZONE PRINCIPALE
st.markdown("# 🚀 Smart Resale AI Predictor")
st.markdown("### Estimation intelligente basée sur le Machine Learning")

# Label non vide pour l'accessibilité, mais caché visuellement
title = st.text_input(
    label="Nom de l'objet", 
    placeholder="ex: Apple iPhone 13 Pro 128GB Bleu",
    label_visibility="collapsed" 
)

if title:
    # --- LOGIQUE ML AVEC FIX POUR LES WARNINGS ---
    num_features = {
        'msrp': input_msrp,
        'has_brand': 1 if any(b in title.upper() for b in ['APPLE', 'SAMSUNG', 'SONY', 'NINTENDO', 'PLAYSTATION', 'LENOVO']) else 0,
        'is_bundle': 1 if is_bundle else 0,
        'title_len': len(title),
        'word_count': len(title.split()),
        'demand_score': demand_score
    }
    
    tfidf = assets['vectorizer']['tfidf']
    svd = assets['vectorizer']['svd']
    text_svd = svd.transform(tfidf.transform([title]))
    text_features = {f'text_svd_{i}': val for i, val in enumerate(text_svd[0])}
    
    # Création du DataFrame avec noms de colonnes explicites pour supprimer les UserWarnings
    full_data = {**num_features, **text_features}
    X_input = pd.DataFrame([full_data])[assets['scaler'].feature_names_in_]
    
    # Transformation et Prédiction
    X_scaled = assets['scaler'].transform(X_input)
    # On force la conversion en DataFrame avec noms pour les modèles
    X_final = pd.DataFrame(X_scaled, columns=assets['scaler'].feature_names_in_)
    
    price = assets['reg'].predict(X_final)[0]
    category = assets['clf'].predict(X_final)[0]
    probs = assets['clf'].predict_proba(X_final)[0]

    # --- AFFICHAGE ---
    st.divider()
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric("Prix Estimé", f"{price:.2f} €", delta=f"{price-input_msrp:.2f} € vs Neuf")
    with m2:
        icon = "🟢" if category == "High" else "🟡" if category == "Mid" else "🟠"
        st.metric("Potentiel Revente", f"{icon} {category}")
    with m3:
        conf = np.max(probs) * 100
        st.metric("Indice de Confiance", f"{conf:.1f}%")

    col_graph, col_nlp = st.columns([1.5, 1])
    
    with col_graph:
        st.markdown("#### 📊 Analyse de Probabilité")
        df_chart = pd.DataFrame({'Segment': ['Low', 'Mid', 'High'], 'Probabilité': probs})
        fig = px.bar(df_chart, x='Segment', y='Probabilité', color='Probabilité', 
                     color_continuous_scale='Blues', text_auto='.2f')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
        st.plotly_chart(fig, width="stretch") # Fix pour l'obsolescence de use_container_width

    with col_nlp:
        st.markdown("#### 🔍 Analyse Sémantique")
        st.write(f"**Marque détectée :** {'✅ Oui' if num_features['has_brand'] else '❌ Non'}")
        st.write(f"**Taille du titre :** {num_features['title_len']} car.")
        
        if conf < 65:
            st.warning("Précision faible. Ajoutez des détails techniques.")
        else:
            st.success("Titre de qualité. Données fiables.")
else:
    st.info("Entrez un titre ci-dessus pour démarrer l'analyse.")