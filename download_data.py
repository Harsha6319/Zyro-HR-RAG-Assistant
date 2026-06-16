import os
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    
    # Authenticate Kaggle via environment variables
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    
    if not username or not key:
        logger.warning(
            "KAGGLE_USERNAME or KAGGLE_KEY not found in environment. "
            "Please copy .env.example to .env and fill in your Kaggle API credentials."
        )
        logger.info("Attempting download without credentials (may fail)...")
    else:
        # kagglehub looks at these environment variables to authenticate
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        logger.info("Kaggle environment variables set successfully.")

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub package is not installed. Please run: pip install -r requirements.txt")
        return

    logger.info("Downloading competition files from 'niat-masterclass-rag-challenge'...")
    try:
        # Download latest version of the competition files
        download_path_str = kagglehub.competition_download("niat-masterclass-rag-challenge")
        download_path = Path(download_path_str)
        logger.info(f"Successfully downloaded files to: {download_path}")

        # Destination data directory
        dest_dir = Path("data")
        dest_dir.mkdir(exist_ok=True)
        
        # Copy files to data/ folder
        copied_files = []
        for file_path in download_path.rglob("*"):
            if file_path.is_file():
                # We want to copy PDFs and CSVs/JSONs to the data directory
                dest_file = dest_dir / file_path.name
                shutil.copy2(file_path, dest_file)
                copied_files.append(file_path.name)
                logger.info(f"Copied: {file_path.name} -> {dest_file}")
        
        logger.info(f"Completed copying {len(copied_files)} files to the local 'data/' directory.")
        
    except Exception as e:
        logger.exception(f"An error occurred during download/extraction: {e}")
        logger.error(
            "Please ensure your Kaggle credentials are correct and you have accepted "
            "the competition rules on the Kaggle website: https://www.kaggle.com/c/niat-masterclass-rag-challenge"
        )

if __name__ == "__main__":
    main()
