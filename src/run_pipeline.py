import time
import logging
import sys
import os

# ==============================================================================
# 1. CONFIGURATION DE L'ENVIRONNEMENT ET DES CHEMINS
# ==============================================================================
# On s'assure que le répertoire 'src' est dans le chemin système pour permettre
# les imports modulaires, peu importe d'où le script est lancé.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Création récursive des dossiers pour les logs et les données si nécessaire
os.makedirs(os.path.join(current_dir, 'data', 'logs'), exist_ok=True)
log_file = os.path.join(current_dir, 'data', 'logs', 'pipeline_run.log')

# ==============================================================================
# 2. IMPORTATION DES ÉTAPES DU PIPELINE
# ==============================================================================
from utils.download_datasets import download_and_extract
from pipeline.step1_prep import run_step1_preparation
from pipeline.step2_features import run_step2_features
from pipeline.step3_training import run_training_pipeline
from pipeline.step4_evaluation import run_step4_evaluation

# ==============================================================================
# 3. CONFIGURATION DU LOGGING (DOUBLE SORTIE : FICHIER ET CONSOLE)
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),      # Archive les logs pour l'audit
        logging.StreamHandler(sys.stdout)   # Affiche l'avancement en temps réel
    ]
)

def check_datasets():
    """
    Étape 0 : Vérification et Acquisition des données.
    Vérifie la présence des fichiers bruts CSV. Si absents, déclenche le 
    téléchargement automatisé depuis Kaggle via l'API.
    """
    ebay_file = 'data/raw/marketing_sample_for_ebay_com-ebay_com_product__20200601_20200831__30k_data.csv'
    amazon_file = 'data/raw/amazon_products.csv'
    
    # Construction des chemins relatifs à la racine du projet
    root_ebay_path = os.path.join(current_dir, '..', ebay_file)
    root_amazon_path = os.path.join(current_dir, '..', amazon_file)

    if not os.path.exists(root_ebay_path) or not os.path.exists(root_amazon_path):
        logging.info("📦 Données brutes manquantes. Lancement du téléchargement Kaggle...")
        # Acquisition des datasets eBay et Amazon
        download_and_extract('promptcloud/ebay-product-listing', os.path.join(current_dir, '..', 'data/raw'))
        download_and_extract('aaronfriasr/amazon-products-dataset', os.path.join(current_dir, '..', 'data/raw'))
    else:
        logging.info("✅ Datasets bruts détectés. Passage à l'étape suivante.")

def run_full_pipeline():
    """
    Exécution complète du workflow CRISP-DM.
    Gère l'enchaînement logique des tâches et mesure le temps d'exécution total.
    """
    start_time = time.time()
    logging.info("🚀 DÉMARRAGE DU PIPELINE COMPLET 'SMART RESALE'")
    
    
    
    try:
        # ÉTAPE 0 : Acquisition (Vérifie que les sources sont prêtes)
        check_datasets()

        # ÉTAPE 1 : Préparation (Nettoyage, Filtrage des Outliers, Fusion Amazon)
        run_step1_preparation()
        
        # ÉTAPE 2 : Feature Engineering (NLP avec SVD, Scaling, Encodage)
        run_step2_features()
        
        # ÉTAPE 3 : Entraînement (Modèles Random Forest Régression & Classification)
        run_training_pipeline()
        
        # ÉTAPE 4 : Évaluation (Génération des métriques et graphiques de diagnostic)
        run_step4_evaluation()
        
        total_time = time.time() - start_time
        logging.info(f"🎉 PIPELINE TERMINÉ AVEC SUCCÈS en {total_time:.2f} secondes !")

    except Exception as e:
        # En cas d'erreur, le traceback complet est capturé dans les logs
        logging.error(f"❌ ÉCHEC DU PIPELINE. Détails : {str(e)}")
        raise

if __name__ == "__main__":
    run_full_pipeline()