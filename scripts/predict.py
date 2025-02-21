import duckdb
import polars as pl
import tensorflow as tf
import joblib
import numpy as np
import os
import constants
from logger import logger

def predict_from_table(table_name: str) -> pl.DataFrame:
    """Loads data from DuckDB and uses the AI model to predict music popularity."""
    
    logger.info(f"🎯 Loading data from DuckDB table: {table_name}...")

    try:
        with duckdb.connect(constants.DUCKDB_PATH, read_only=False) as con:
            # Query the database
            query = f"""
                SELECT id, danceability, energy, tempo, valence, loudness, 
                       speechiness, instrumentalness, acousticness, mode, key, duration_ms 
                FROM {table_name}
            """
            arrow_table = con.execute(query).fetch_arrow_table()
            df = pl.from_arrow(arrow_table)

            # Check if the table is empty
            if df.is_empty():
                logger.error("❌ No data found in the table.")
                raise ValueError("The table is empty, no data to predict.")

            # Separate track IDs from features
            track_info = df.select(['id'])
            features = df.drop(['id'])

            # Load AI model and scaler
            logger.info("🧠 Loading AI model...")

            if not os.path.exists(constants.MODEL_PATH):
                logger.error(f"❌ Model file not found: {constants.MODEL_PATH}")
                raise FileNotFoundError(f"Model file not found: {constants.MODEL_PATH}")

            if not os.path.exists(constants.SCALER_PATH):
                logger.error(f"❌ Scaler file not found: {constants.SCALER_PATH}")
                raise FileNotFoundError(f"Scaler file not found: {constants.SCALER_PATH}")

            model = tf.keras.models.load_model(constants.MODEL_PATH)
            scaler = joblib.load(constants.SCALER_PATH)

            # Normalize feature data
            features_scaled = scaler.transform(features)

            # Make predictions
            predictions = model.predict(features_scaled)
            predicted_classes = np.argmax(predictions, axis=1)

            # Add predictions to DataFrame
            results = track_info.with_columns(pl.Series("class", predicted_classes))
            logger.info("✅ Predictions complete! Updating database...")

            # Ensure the 'class' column exists in the table
            con.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS class INTEGER;")

            # Batch update predictions in the database
            con.executemany(
                f"UPDATE {table_name} SET class = ? WHERE id = ?",
                [(int(row["class"]), row["id"]) for row in results.iter_rows(named=True)]
            )

            logger.info("✅ Database successfully updated with predictions.")

            return results

    except duckdb.Error as e:
        logger.error(f"❌ DuckDB error: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"❌ Missing file: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in predictFromTable: {e}")
        raise

if __name__ == "__main__":
    predict_from_table(constants.PREDICT_TABLE_NAME)
