import os
from dotenv import load_dotenv

# Load and set environment variables BEFORE importing KaggleApi
load_dotenv()

# Set credentials in environment
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

# Verify credentials are set
if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
    raise ValueError("❌ KAGGLE_USERNAME or KAGGLE_KEY not found in .env file!")

print(f"✓ Credentials loaded for: {os.environ['KAGGLE_USERNAME']}")

# NOW import KaggleApi after credentials are set
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract(dataset_slug, download_path='data/raw'):
    """
    Download and extract Kaggle dataset
    
    Args:
        dataset_slug: Dataset identifier (e.g., 'promptcloud/ebay-product-listing')
        download_path: Where to save files
    """
    # Create directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    print(f"\n📥 Downloading {dataset_slug}...")
    
    # Initialize and authenticate API
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
    
    print(f"✓ {dataset_slug} downloaded and extracted to {download_path}")

def main():
    """Download both datasets"""
    try:
        # Download eBay dataset
        download_and_extract('promptcloud/ebay-product-listing')
        
        # Download Amazon dataset
        download_and_extract('aaronfriasr/amazon-products-dataset')
        
        # List downloaded files
        print("\n" + "="*50)
        print("📁 Files downloaded in data/raw/:")
        print("="*50)
        
        files = [f for f in os.listdir('data/raw') if f.endswith('.csv')]
        for file in files:
            file_path = os.path.join('data/raw', file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ✓ {file} ({size_mb:.2f} MB)")
        
        print("\n🎉 All datasets downloaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()