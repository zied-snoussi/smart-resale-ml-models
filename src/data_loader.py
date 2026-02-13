import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

# Load environment variables
load_dotenv()

def download_kaggle_dataset(dataset_name, download_path='data/raw'):
    """
    Download dataset from Kaggle using API credentials from .env
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'username/dataset-name')
        download_path: Where to save the dataset
    
    Example:
        download_kaggle_dataset('annavictoria/speed-dating-experiment')
    """
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Download dataset
    print(f"Downloading {dataset_name}...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"Dataset downloaded to {download_path}")

if __name__ == "__main__":
    # Example: Download your dataset
    download_kaggle_dataset('your-dataset-here')