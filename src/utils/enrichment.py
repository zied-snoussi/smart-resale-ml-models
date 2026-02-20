import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from utils.load_data import load_amazon_products, load_amazon_categories

def clean_text_key(text):
    """
    Normalise le texte pour optimiser la correspondance sémantique.
    Supprime les caractères spéciaux, les parenthèses et convertit en minuscules.
    """
    if not isinstance(text, str):
        return ""
    
    # Suppression du contenu entre parenthèses (ex: "(Renewed)", "(Refurbished)")
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Conservation uniquement des caractères alphanumériques et des espaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    
    # Nettoyage des espaces doubles ou superflus
    return " ".join(text.split())

def enrich_ebay_with_amazon(df_ebay):
    """
    Enrichit le dataset eBay en faisant correspondre les produits avec le catalogue Amazon.
    Utilise TF-IDF + k-Nearest Neighbors pour trouver le prix de référence (MSRP).
    """
    print("\n" + "~"*40)
    print("✨ DÉMARRAGE DE L'ENRICHISSEMENT (MATCHING SÉMANTIQUE)")
    print("~"*40)

    # 1. Chargement et filtrage du catalogue Amazon
    print("\n🔍 Chargement du catalogue Amazon...")
    cats = load_amazon_categories()
    
    # Ciblage exclusif des catégories technologiques pour réduire le "bruit" dans les données
    tech_keywords = ['Electronics', 'Computer', 'Phone', 'Tablet', 'Laptop', 'Camera', 'Headphone']
    tech_cats = cats[cats['category_name'].str.contains('|'.join(tech_keywords), case=False, na=False)]
    valid_cat_ids = set(tech_cats['id'].unique())
    
    df_amazon = load_amazon_products()
    df_amz = df_amazon[df_amazon['category_id'].isin(valid_cat_ids)].copy()
    
    # On ne conserve que les produits avec un prix valide pour servir de référence
    df_amz = df_amz[df_amz['price'] > 0].reset_index(drop=True)
    df_amz['clean_title'] = df_amz['title'].apply(clean_text_key)
    
    print(f"   Catalogue de référence : {len(df_amz):,} produits Tech")

    # 2. Construction de l'index de recherche vectoriel (TF-IDF)
    print("\n⚙️ Construction de l'index de recherche (TF-IDF)...")
    vectorizer = TfidfVectorizer(min_df=1, analyzer='word', stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_amz['clean_title'])
    
    # Utilisation de k-NN avec la distance cosinus pour mesurer la similarité entre titres
    nn_model = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=-1)
    nn_model.fit(tfidf_matrix)
    print("   Index de recherche généré avec succès.")

    # 3. Préparation des requêtes eBay
    print("\n🤝 Appariement des bases de données...")
    
    def make_query(row):
        """Génère une chaîne de recherche propre à partir de la marque et du modèle."""
        brand = str(row.get('Manufacturer', '')).strip()
        model = str(row.get('Model Name', '')).strip()
        # Si le modèle est précis, on l'utilise en priorité avec la marque
        if len(model) > 3 and model.lower() != 'nan':
            return clean_text_key(f"{brand} {model}")
        # Sinon, on se rabat sur le titre complet de l'annonce
        return clean_text_key(row['Title'])
    
    print("   Génération des requêtes depuis les données eBay...")
    ebay_queries = df_ebay.apply(make_query, axis=1).tolist()
    ebay_tfidf = vectorizer.transform(ebay_queries)
    
    # 4. Exécution de la recherche de similarité
    print("   Exécution de la recherche de similarité...")
    distances, indices = nn_model.kneighbors(ebay_tfidf)
    
    # 5. Traitement et extraction des résultats
    matched_features = {
        'original_price': [], 'matched_asin': [],
        'match_score': [], 'match_title': [], 'demand_score': []
    }
    
    # Seuil de tolérance : plus la distance est proche de 0, plus le match est exact
    THRESHOLD_DISTANCE = 0.45 
    
    for i, distance in enumerate(distances):
        dist = distance[0]
        idx = indices[i][0]
        
        if dist < THRESHOLD_DISTANCE:
            amz_row = df_amz.iloc[idx]
            
            # Priorité au prix catalogue (MSRP), sinon prix actuel Amazon
            orig = amz_row.get('listPrice', 0)
            cur_price = amz_row.get('price', 0)
            final_ref_price = orig if orig > 0 else (cur_price if cur_price > 0 else 0)
                
            matched_features['original_price'].append(final_ref_price)
            matched_features['matched_asin'].append(amz_row['asin'])
            matched_features['match_score'].append(1 - dist) # Score de confiance
            matched_features['match_title'].append(amz_row['title'])
            matched_features['demand_score'].append(amz_row.get('boughtInLastMonth', 0))
        else:
            # Valeurs par défaut en cas d'absence de correspondance satisfaisante
            for key in matched_features: matched_features[key].append(0 if 'score' in key or 'price' in key else None)

    # 6. Intégration dans le DataFrame eBay original
    df_ebay['original_price'] = matched_features['original_price']
    df_ebay['matched_asin'] = matched_features['matched_asin']
    df_ebay['match_confidence'] = matched_features['match_score']
    df_ebay['amazon_demand'] = matched_features['demand_score']
    
    # Nettoyage du prix eBay (conversion en float) si nécessaire
    if 'price_cleaned' not in df_ebay.columns:
         df_ebay['price_cleaned'] = df_ebay['Price'].apply(lambda x: float(re.sub(r'[^\d.]', '', str(x))) if pd.notna(x) else 0)

    # --- FILTRE DE COHÉRENCE (SANITY FILTER) ---
    # On élimine les matchs illogiques (ex: prix occasion > 1.5x prix neuf Amazon)
    # Cela arrive souvent en cas de mauvais matching (accessoire matché avec un pack console)
    mask_anomaly = (df_ebay['original_price'] > 0) & (df_ebay['price_cleaned'] > (df_ebay['original_price'] * 1.5))
    anomaly_count = mask_anomaly.sum()
    
    if anomaly_count > 0:
        print(f"\n🧹 Filtre de cohérence : Suppression de {anomaly_count:,} anomalies (Prix Occasion > 1.5x Prix Neuf)")
        df_ebay.loc[mask_anomaly, ['original_price', 'matched_asin', 'match_confidence']] = [0, None, 0]

    # Calcul du taux de dépréciation (entre 0 et 1)
    mask = (df_ebay['original_price'] > 0)
    df_ebay['depreciation_pct'] = 0.0
    df_ebay.loc[mask, 'depreciation_pct'] = (
        (df_ebay.loc[mask, 'original_price'] - df_ebay.loc[mask, 'price_cleaned']) 
        / df_ebay.loc[mask, 'original_price']
    ).clip(0, 1)

    # Statistiques finales
    matches_found = (df_ebay['original_price'] > 0).sum()
    print(f"\n✅ Enrichissement terminé !")
    print(f"   Matches trouvés : {matches_found:,} ({matches_found/len(df_ebay)*100:.1f}%)")
    
    return df_ebay