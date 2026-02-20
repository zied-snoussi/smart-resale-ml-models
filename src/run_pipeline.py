import time
import logging
import sys
import os

# 1. SETUP ENVIRONMENT & PATHS
# Ensure we are working relative to the 'src' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Ensure the log and data directories exist BEFORE logging starts
os.makedirs(os.path.join(current_dir, 'data', 'logs'), exist_ok=True)
log_file = os.path.join(current_dir, 'data', 'logs', 'pipeline_run.log')

# 2. IMPORT STEPS
# Import your utility to download data
from utils.download_datasets import download_and_extract
# Import the pipeline steps
from pipeline.step1_prep import run_step1_preparation
from pipeline.step2_features import run_step2_features
from pipeline.step3_training import run_training_pipeline
from pipeline.step4_evaluation import run_step4_evaluation

# 3. CONFIGURE LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def check_datasets():
    """Step 0: Verify raw data exists, otherwise download it."""
    ebay_file = 'data/raw/marketing_sample_for_ebay_com-ebay_com_product__20200601_20200831__30k_data.csv'
    amazon_file = 'data/raw/amazon_products.csv'
    
    # Path relative to the root project folder
    root_ebay_path = os.path.join(current_dir, '..', ebay_file)
    root_amazon_path = os.path.join(current_dir, '..', amazon_file)

    if not os.path.exists(root_ebay_path) or not os.path.exists(root_amazon_path):
        logging.info("📦 Raw data missing. Starting download from Kaggle...")
        download_and_extract('promptcloud/ebay-product-listing', os.path.join(current_dir, '..', 'data/raw'))
        download_and_extract('aaronfriasr/amazon-products-dataset', os.path.join(current_dir, '..', 'data/raw'))
    else:
        logging.info("✅ Raw datasets already present. Skipping download.")

def run_full_pipeline():
    """The complete CRISP-DM workflow execution."""
    start_time = time.time()
    logging.info("🚀 STARTING SMART RESALE ML PIPELINE")
    
    try:
        # STEP 0: Data Acquisition
        check_datasets()

        # STEP 1: Data Preparation (Cleaning & Enrichment)
        run_step1_preparation()
        
        # STEP 2: Feature Engineering (NLP & Scaling)
        run_step2_features()
        
        # STEP 3: Training (Regression & Classification)
        run_training_pipeline()
        
        # STEP 4: Evaluation (Metrics & Visualization)
        run_step4_evaluation()
        
        total_time = time.time() - start_time
        logging.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f} seconds!")

    except Exception as e:
        logging.error(f"❌ PIPELINE FAILED. Error: {str(e)}")
        raise

if __name__ == "__main__":
    run_full_pipeline()