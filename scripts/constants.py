import os

# Environment setting (e.g., "local", "production")
ENV = os.getenv("ENV", "local")

# Automatically find the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Subdirectories
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "train")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DATA_PREDICT_DIR = os.path.join(DATA_DIR, "predict")

# DuckDB Database
DUCKDB_FILENAME = "music_analysis.duckdb"
DUCKDB_PATH = os.path.join(DATA_PROCESSED_DIR, DUCKDB_FILENAME)

# Train data
TRAIN_TRACKS_URI = "yamaerenay/spotify-dataset-19212020-600k-tracks"
TRAIN_FILENAME = "tracks.csv"
TRAIN_TRACKS_FILENAME = os.path.join(DATA_TRAIN_DIR, TRAIN_FILENAME)
TRAIN_TABLE_NAME = "train_tracks"

# Predict data
PREDICT_FILENAME = "new_tracks.csv"
PREDICT_TRACKS_FILENAME = os.path.join(DATA_PREDICT_DIR, PREDICT_FILENAME)
PREDICT_TABLE_NAME = "predict_tracks"

# Neural Network model paths
NN_MODEL_FILENAME = "nn_model.keras"
SCALER_FILENAME = "scaler.pkl"

MODEL_PATH = os.path.join(MODEL_DIR, NN_MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)

# Logging settings
LOG_FILE = os.path.join(LOG_DIR, "app.log")
