from prefect import flow, task
import constants
import extract
import transform
import load
import predict
import visualize
from logger import logger

@task
def extract_data():
    """Extracts dataset from CSV file for prediction."""
    logger.info("📥 Extracting data from CSV file...")
    
    df = extract.extract_csv(constants.PREDICT_TRACKS_FILENAME)
    
    if df is None or df.is_empty():
        logger.error("❌ Extracted dataset is empty!")
        raise ValueError("Extracted dataset is empty!")

    logger.info("✅ Data extraction completed successfully.")
    return df

@task
def transform_data(df):
    """Cleans and transforms the dataset before prediction."""
    logger.info("🔄 Transforming data...")

    df_cleaned = transform.transform(df)

    if df_cleaned.is_empty():
        logger.error("❌ Transformed dataset is empty!")
        raise ValueError("Transformed dataset is empty!")

    logger.info("✅ Data transformation completed successfully.")
    return df_cleaned

@task
def load_data(df):
    """Loads transformed data into DuckDB for prediction."""
    logger.info("📥 Loading data into DuckDB...")

    try:
        load.load(df, constants.PREDICT_TABLE_NAME)
        logger.info("✅ Data successfully loaded into DuckDB!")
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

@task
def predict_data():
    """Runs predictions using the trained model."""
    logger.info("🤖 Running predictions...")

    try:
        predict.predict_from_table(constants.PREDICT_TABLE_NAME)
        logger.info("✅ Predictions completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        raise

@task
def visualize_data():
    """Generates visualizations for the predictions."""
    logger.info("📊 Generating visualizations...")

    try:
        visualize.visualize_data(constants.PREDICT_TABLE_NAME)
        logger.info("✅ Visualizations generated successfully!")
    except Exception as e:
        logger.error(f"❌ Error generating visualizations: {e}")
        raise

@flow
def music_pipeline():
    """Prefect pipeline for extracting, transforming, loading, predicting, and visualizing data."""
    df = extract_data()
    df_cleaned = transform_data(df)
    load_data(df_cleaned)
    predict_data()
    visualize_data()

if __name__ == "__main__":    
    music_pipeline()
