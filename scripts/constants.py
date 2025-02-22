import os

# Environment setting (e.g., "local", "production")
ENV = os.getenv("ENV", "local")

# Automatically find the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# DuckDB Database
DUCKDB_FILENAME = "music_analysis.duckdb"
DUCKDB_PATH = os.path.join(DATA_DIR, DUCKDB_FILENAME)

# Raw data
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_RAW_KAGGLE_URI = "yamaerenay/spotify-dataset-19212020-600k-tracks"
DATA_RAW_FILENAME = os.path.join(DATA_RAW_DIR, "tracks.csv")
DATA_RAW_TABLENAME = "raw"

# Split data
DATA_TRAIN_TABLENAME = "train"
DATA_PREDICT_TABLENAME = "predict"

# Neural Network model
NN_MODEL_FILENAME = "nn_model.keras"
SCALER_FILENAME = "scaler.pkl"

MODEL_PATH = os.path.join(MODEL_DIR, NN_MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)

## Defines popularity classes
CLASSES = {
    0: "🧊 Ice-cold (0-30)",  
    1: "🔥 Lukewarm (31-69)",  
    2: "🚀 Blazing Hot (70-100)"  
}

## Hyperparameters
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 0.2
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.2

