import duckdb
import polars as pl
import tensorflow as tf
import joblib
import os
import numpy as np

DATA_DIR = "data"
MODEL_DIR = "models"
DUCKDB_PATH = os.path.join(DATA_DIR, "music_analysis.duckdb")
MODEL_PATH = os.path.join(MODEL_DIR, "nn_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def predict(table_name):
    """Carica i dati da DuckDB e utilizza il modello AI per predire la popolarità."""
    print("🎯 Loading data from DuckDB...")
    
    with duckdb.connect(DUCKDB_PATH, read_only=False) as con:
        arrow_table = con.execute(f"SELECT id, danceability, energy, tempo, valence, loudness, speechiness, instrumentalness, acousticness, mode, key, duration_ms FROM {table_name}").fetch_arrow_table()
        df = pl.from_arrow(arrow_table)
        
        track_info = df.select(['id'])
        features = df.drop(['id'])

        print("🧠 Loading AI model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        features_scaled = scaler.transform(features)

        predictions = model.predict(features_scaled)

        predicted_classes = np.argmax(predictions, axis=1)
        results = track_info.with_columns(pl.Series("class", predicted_classes))

        print("✅ Predictions complete! Updating database...")
        con.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS class INTEGER;")
        
        for row in results.iter_rows(named=True):
            con.execute(f"UPDATE {table_name} SET class = {row['class']} WHERE id = '{row['id']}'")
                
    return results

if __name__ == "__main__":
    predict("predict_tracks")
