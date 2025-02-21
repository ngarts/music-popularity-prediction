import os
import polars as pl
import subprocess
from logger import logger

def download_dataset(uri, path):
    """Downloads dataset from Kaggle if it is not already present."""
    
    if not os.path.exists(path):
        os.makedirs(path)

        logger.info("📥 Downloading dataset from Kaggle...")
        try:
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", uri,
                "-p", path,
                "--unzip"
            ], check=True)
            logger.info("✅ Dataset downloaded successfully!")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error downloading dataset: {e}")
        except FileNotFoundError:
            logger.error("❌ Kaggle CLI not found. Make sure it is installed and configured.")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
    else:
        logger.info("✅ Dataset already exists, skipping download.")

def extract_csv(filename):
    """Extracts data from the CSV file."""
    
    try:
        logger.info("📥 Extracting data from CSV...")
        df = pl.read_csv(filename)
        logger.info("✅ Data extracted successfully!")
        return df
    except FileNotFoundError:
        logger.error(f"❌ File not found: {filename}")
        return None
    except Exception as e:
        logger.error(f"❌ Error reading CSV file: {e}")
        return None

def extract_kaggle(uri, path, filename):
    """Downloads and extracts data from a Kaggle dataset."""
    
    download_dataset(uri, path)
    
    df = extract_csv(filename)
    
    if df is not None:
        logger.info("✅ Data extraction completed successfully.")
    else:
        logger.warning("⚠️ Data extraction failed.")
    
    return df

