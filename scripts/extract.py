import os
import polars as pl
import subprocess

DATA_DIR = "data"

def download_dataset(uri, filename):
    """Scarica il dataset da Kaggle se non è già presente."""
    
    if not os.path.exists(filename):
        print("📥 Downloading dataset from Kaggle...")
        try:
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", uri,
                "-p", DATA_DIR,
                "--unzip"
            ], check=True)
            print("✅ Dataset downloaded successfully!")
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
    else:
        print("✅ Dataset already exists, skipping download.")

def extract_csv(filename):
    
    print("📥 Extracting data...")
    df = pl.read_csv(filename)
    return df

def extract_kaggle(uri, filename):
    """Estrae i dati dal file CSV dopo averlo scaricato."""
    download_dataset(uri, filename)
    
    df = extract_csv(filename)
    return df

