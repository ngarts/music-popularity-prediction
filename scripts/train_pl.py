from prefect import flow, task
import constants
import extract
import transform
import load
import train
from logger import logger

@task
def extract_data():
    """Extracts dataset from Kaggle."""
    logger.info("📥 Extracting data from Kaggle...")
    df = extract.extract_kaggle(constants.TRAIN_TRACKS_URI, constants.DATA_TRAIN_DIR, constants.TRAIN_TRACKS_FILENAME)
    
    if df is None or df.is_empty():
        logger.error("❌ Extracted dataset is empty!")
        raise ValueError("Extracted dataset is empty!")

    logger.info("✅ Data extraction completed successfully.")
    return df

@task
def transform_data(df):
    """Cleans and transforms the dataset."""
    logger.info("🔄 Transforming data...")
    df_cleaned = transform.transform(df)

    if df_cleaned.is_empty():
        logger.error("❌ Transformed dataset is empty!")
        raise ValueError("Transformed dataset is empty!")

    logger.info("✅ Data transformation completed successfully.")
    return df_cleaned

@task
def load_data(df):
    """Loads cleaned data into DuckDB."""
    logger.info("📥 Loading data into DuckDB...")
    try:
        load.load(df, constants.TRAIN_TABLE_NAME)
        logger.info("✅ Data successfully loaded into DuckDB!")
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

@task
def train_model():
    """Trains the machine learning model."""
    logger.info("🚀 Starting model training...")
    try:
        train.train_from_table(constants.TRAIN_TABLE_NAME)
        logger.info("✅ Model training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        raise

@flow
def music_pipeline():
    """Prefect pipeline for extracting, transforming, loading, and training the model."""
    df = extract_data()
    df_cleaned = transform_data(df)
    load_data(df_cleaned)
    train_model()

if __name__ == "__main__":
    music_pipeline()
