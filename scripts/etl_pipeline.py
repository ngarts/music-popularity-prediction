import constants
import os
import polars as pl
import kaggle
import duckdb
from sklearn.model_selection import train_test_split
from prefect import flow, task
from prefect.logging import get_run_logger

@task(name="etl-download-dataset")
def download_dataset(uri, path):
    """Downloads dataset from Kaggle if it is not already present."""

    logger = get_run_logger() 

    if not os.path.exists(path):

        os.makedirs(path, exist_ok=True)
    
        logger.info("📥 Downloading dataset from Kaggle...")
        try:
            kaggle.api.dataset_download_files(uri, path=path, unzip=True)
            logger.info("✅ Dataset downloaded successfully!")
        except Exception as e:
            logger.error(f"❌ Error downloading dataset: {e}")
            raise
    else:
        logger.info("✅ Dataset already exists, skipping download.")

@task(name="etl-extract-csv")
def extract_csv(filename):
    """Extracts data from CSV file."""
    
    logger = get_run_logger()

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

@task(name="etl-transform")
def transform(df):
    """Cleans and transforms the extracted dataset."""
    
    logger = get_run_logger()
    logger.info("🔄 Transforming data...")

    selected_columns = [
        "id", "name", "artists", "danceability", "energy", "tempo",
        "valence", "loudness", "speechiness", "instrumentalness",
        "acousticness", "mode", "key", "duration_ms", "popularity"
    ]
    
    # Check for missing columns
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"❌ ERROR: Missing columns in dataset: {missing_columns}")
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Select only necessary columns and drop nulls
    df_cleaned = df.select(selected_columns).drop_nulls()

    # Convert popularity into 3 classes
    def popularity_to_class(popularity):
        if popularity <= 30:
            return 0  # Ice-cold 🧊
        elif popularity <= 69:
            return 1  # Lukewarm 🔥
        else:
            return 2  # Blazing Hot 🚀

    df_cleaned = df_cleaned.with_columns(
        df_cleaned["popularity"].map_elements(popularity_to_class).alias("popularity_class")
    )

    logger.info("✅ Data transformation completed successfully.")
    return df_cleaned

@task(name="etl-split-dataset")
def split_dataset(df):
    """Splits dataset into Train and Predict sets."""

    logger = get_run_logger()
    logger.info("📊 Splitting dataset into Train and Predict...")

    if "popularity" not in df.columns:
        raise ValueError("🚨 Column 'popularity' not found in dataset!")
    
    # Get class distribution
    class_counts = df["popularity_class"].value_counts()

     # Convert value_counts() result into a dictionary
    class_counts_dict = dict(zip(class_counts["popularity_class"].to_list(), class_counts["count"].to_list()))
    logger.info(f"📊 Class distribution before split: {class_counts_dict}")

    # Remove rare classes (less than 2 samples)
    rare_classes = [cls for cls, count in class_counts_dict.items() if count < 2]

    if rare_classes:
        logger.warning(f"⚠️ The following classes have less than 2 samples and will be removed: {rare_classes}")
        df = df.filter(~df["popularity_class"].is_in(rare_classes))

    # Conta il numero di classi uniche
    num_unique_classes = df["popularity_class"].n_unique()
    stratify = df["popularity_class"].to_pandas() if num_unique_classes > 1 else None

    # Split dataset (90% Train, 10% Predict) stratified by class
    train_df, predict_df = train_test_split(
        df.to_pandas(), test_size=0.1, random_state=42, stratify=stratify
    )

    # Prediction dataset DOES NOT need 'popularity' column
    predict_df = predict_df.drop(columns=["popularity"])

    logger.info("✅ Dataset successfully split into Train and Predict!")
    return train_df, predict_df

@task(name="etl-load-into-DuckDB")
def load(df, table_name):
    """Loads cleaned data into DuckDB."""

    logger = get_run_logger()
    logger.info(f"📥 Loading data into DuckDB table: {table_name}...")

    try:
        with duckdb.connect(constants.DUCKDB_PATH, read_only=False) as con:
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

        logger.info(f"✅ Data successfully loaded into DuckDB table: {table_name}!")
    except duckdb.Error as e:
        logger.error(f"❌ Error loading data into DuckDB: {e}")
        raise

@flow(name="etl-pipeline")
def etl_pipeline():
    """Runs the full ETL pipeline: Extract, Transform, Load"""

    logger = get_run_logger()
    logger.info("🚀 Starting ETL pipeline...")

    # Extract data
    download_dataset(constants.DATA_RAW_KAGGLE_URI, constants.DATA_RAW_DIR)
    
    df = extract_csv(constants.DATA_RAW_FILENAME)

    if df is None:
        logger.error("❌ ETL Pipeline failed: No data extracted!")
        return
    else:
        logger.info(f"📊 Raw dataset contains {df.height} rows and {df.width} columns.")

    # Transform data
    cleaned_df = transform(df)

    logger.info(f"📊 Cleaned raw dataset contains {cleaned_df.height} rows and {cleaned_df.width} columns.")

    # Split into train and predict sets
    train_df, predict_df = split_dataset(cleaned_df)
    
    logger.info(f"📊 Train dataset contains {len(train_df)} rows and {len(train_df.columns)} columns.")
    logger.info(f"📊 Predict dataset contains {len(predict_df)} rows and {len(predict_df.columns)} columns.")

    # Load data into DuckDB
    load(train_df, constants.DATA_TRAIN_TABLENAME)
    load(predict_df, constants.DATA_PREDICT_TABLENAME)

    logger.info("✅ ETL pipeline completed successfully!")

if __name__ == "__main__":
    etl_pipeline()
