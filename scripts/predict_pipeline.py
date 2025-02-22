from prefect import flow, task
from prefect.logging import get_run_logger
import duckdb
import polars as pl
import tensorflow as tf
import joblib
import numpy as np
import os
import constants
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

@task(name="predict-load-data")
def load_data(table_name: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Loads data from DuckDB table."""
    
    logger = get_run_logger() 
    logger.info(f"🎯 Loading data from DuckDB table: {table_name}...")

    try:
        with duckdb.connect(constants.DUCKDB_PATH, read_only=False) as con:
            query = f"""
                SELECT id, name, artists, danceability, energy, tempo, valence, loudness, 
                       speechiness, instrumentalness, acousticness, mode, key, duration_ms 
                FROM {table_name}
            """
            arrow_table = con.execute(query).fetch_arrow_table()
            df = pl.from_arrow(arrow_table)

            if df.is_empty():
                logger.error("❌ No data found in the table.")
                raise ValueError("The table is empty, no data to predict.")

            track_info = df.select(['id', 'name', 'artists'])
            features = df.drop(['id', 'name', 'artists'])

            return track_info, features

    except duckdb.Error as e:
        logger.error(f"❌ DuckDB error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in load_data: {e}")
        raise

@task(name="predict-data")
def predict_data(track_info, features, table_name: str) -> pl.DataFrame:
    """Predicts music popularity using the AI model."""
    
    logger = get_run_logger() 

    try:
        logger.info("🧠 Loading AI model...")

        if not os.path.exists(constants.MODEL_PATH):
            logger.error(f"❌ Model file not found: {constants.MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {constants.MODEL_PATH}")

        if not os.path.exists(constants.SCALER_PATH):
            logger.error(f"❌ Scaler file not found: {constants.SCALER_PATH}")
            raise FileNotFoundError(f"Scaler file not found: {constants.SCALER_PATH}")

        model = tf.keras.models.load_model(constants.MODEL_PATH)
        scaler = joblib.load(constants.SCALER_PATH)

        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        predicted_classes = np.argmax(predictions, axis=1)

        results = track_info.with_columns(pl.Series("class", predicted_classes))
        logger.info("✅ Predictions complete! Updating database...")

        with duckdb.connect(constants.DUCKDB_PATH, read_only=False) as con:
            con.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS class INTEGER;")
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
        logger.error(f"❌ Unexpected error in predict_data: {e}")
        raise

@task(name="visualize-results")
def visualize_data(table_name: str):
    """Visualizes predicted data for analysis."""
    
    logger = get_run_logger()
    logger.info("📊 Loading predicted data from DuckDB for visualization...")

    try:
        with duckdb.connect(constants.DUCKDB_PATH, read_only=True) as con:
            query = f"""
                SELECT name, artists, class, danceability, energy, tempo 
                FROM {table_name}
            """
            arrow_table = con.execute(query).fetch_arrow_table()
            df = pl.from_arrow(arrow_table)

            if df.is_empty():
                logger.error("❌ No data found in the table.")
                raise ValueError("The table is empty, no data to visualize.")

            logger.info("✅ Data successfully loaded.")
            logger.info(f"\n{df.head()}")  

            # 🔹 1. Bar chart of popularity class distribution
            plt.figure(figsize=(8, 5))
            sns.countplot(y=df["class"], order=df["class"].value_counts().sort("class")["class"].to_list())
            plt.xlabel("Number of Tracks")
            plt.ylabel("Popularity Class")
            plt.title("Distribution of Tracks by Popularity Class")
            plt.tight_layout()
            plt.show(block=False)

            # 🔹 2. Scatter plot of Danceability vs Energy
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x="danceability", y="energy", hue="class", palette="viridis", alpha=0.7)
            plt.xlabel("Danceability")
            plt.ylabel("Energy")
            plt.title("Danceability vs Energy by Popularity Class")
            plt.tight_layout()
            plt.show(block=False)

            # 🔹 3. WordCloud of track names
    
            # Filter only "Blazing Hot" songs
            df_hits = df.filter(df["class"] == 2)

            # Generate WordCloud only for hits
            text = " ".join(df_hits["name"].to_list())

            if text.strip():  
                wordcloud = WordCloud(width=800, height=400, background_color="black").generate(text)

                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title("WordCloud of Most Popular Track Titles")
                plt.tight_layout()
                plt.show()
            else:
                logger.warning("⚠️ No valid track names found for WordCloud visualization.")

    except duckdb.Error as e:
        logger.error(f"❌ DuckDB error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in visualize_data: {e}")
        raise

@flow(name="predict-pipeline")
def predict_pipeline():
    """Prefect pipeline for loading, predicting, and visualizing data."""
    track_info, features = load_data(constants.DATA_PREDICT_TABLENAME)
    predict_data(track_info, features, constants.DATA_PREDICT_TABLENAME)
    visualize_data(constants.DATA_PREDICT_TABLENAME)

if __name__ == "__main__":
    predict_pipeline()
