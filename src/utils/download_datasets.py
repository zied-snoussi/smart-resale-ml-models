import os
import traceback
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURATION DE L'ENVIRONNEMENT
# ==============================================================================

# Charger les variables d'environnement depuis le fichier .env
# Cette étape est cruciale pour la sécurité : ne jamais coder les clés API en dur.
load_dotenv()

# Injection des identifiants Kaggle dans l'environnement système
# L'API Kaggle recherche automatiquement ces variables spécifiques au démarrage.
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

# Validation rigoureuse des informations d'authentification
if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
    raise ValueError("❌ Erreur : KAGGLE_USERNAME ou KAGGLE_KEY introuvable dans le fichier .env !")

print(f"✓ Identifiants chargés pour l'utilisateur : {os.environ['KAGGLE_USERNAME']}")

# Importation différée de l'API après la configuration des variables système
from kaggle.api.kaggle_api_extended import KaggleApi

# ==============================================================================
# FONCTIONS CORE
# ==============================================================================

def download_and_extract(dataset_slug, download_path='data/raw'):
    """
    Télécharge et extrait automatiquement un jeu de données Kaggle.
    
    Args:
        dataset_slug (str): Identifiant unique du dataset (ex: 'user/dataset-name')
        download_path (str): Répertoire de destination pour le stockage local.
    """
    # Garantir l'existence du dossier de destination (création si nécessaire)
    os.makedirs(download_path, exist_ok=True)
    
    print(f"\n📥 Téléchargement du jeu de données : {dataset_slug}...")
    
    # Initialisation de l'instance API et authentification via les variables d'environnement
    api = KaggleApi()
    api.authenticate()
    
    # Téléchargement : unzip=True permet de décompresser directement les fichiers CSV
    api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
    
    print(f"✓ {dataset_slug} téléchargé et extrait avec succès dans : {download_path}")

def main():
    """
    Point d'entrée principal du script d'ingestion.
    Gère le téléchargement séquentiel et affiche un résumé des fichiers récupérés.
    """
    try:
        # Étape 1 : Acquisition des données sources eBay
        download_and_extract('promptcloud/ebay-product-listing')
        
        # Étape 2 : Acquisition des données de référence Amazon (MSRP)
        download_and_extract('aaronfriasr/amazon-products-dataset')
        
        # --- RÉSUMÉ DE L'INGESTION ---
        print("\n" + "="*50)
        print("📁 RÉCAPITULATIF DES FICHIERS (data/raw/) :")
        print("="*50)
        
        # Parcours et analyse des fichiers CSV téléchargés pour validation d'intégrité
        files = [f for f in os.listdir('data/raw') if f.endswith('.csv')]
        for file in files:
            file_path = os.path.join('data/raw', file)
            # Conversion de la taille en Mo pour une meilleure lisibilité
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✓ {file:<30} | Taille : {size_mb:.2f} MB")
        
        print("\n🎉 Tous les jeux de données ont été récupérés avec succès !")
        
    except Exception as e:
        # Capture et affichage détaillé de l'erreur pour faciliter le débogage (Traceback)
        print(f"\n❌ Erreur lors de l'exécution : {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()