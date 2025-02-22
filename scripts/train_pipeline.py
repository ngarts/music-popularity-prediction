from prefect import flow, task
from prefect.logging import get_run_logger
import constants
import numpy as np
import duckdb
import polars as pl
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.layers import Dense, Dropout

@task(name="train-load-data")
def load_data(table_name: str) -> pl.DataFrame:
    """Loads and dataset from DuckDB."""

    logger = get_run_logger() 
    logger.info(f"📥 Loading dataset from DuckDB table: {table_name}...")

    try:
        
        with duckdb.connect(constants.DUCKDB_PATH, read_only=True) as con:

            query = f"""
                SELECT danceability, energy, tempo, valence, loudness, 
                    speechiness, instrumentalness, acousticness, mode, key, duration_ms, popularity 
                FROM {table_name}
            """
            
            arrow_df = con.execute(query).fetch_arrow_table()
            df = pl.from_arrow(arrow_df)

            if df.is_empty():
                logger.error(f"❌ No data found in table: {table_name}")
                raise ValueError("Dataset is empty!")
            
            return df
        
    except duckdb.Error as e:
        logger.error(f"❌ DuckDB error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in load_data: {e}")
        raise

@task(name="train-transform-data")
def transform_data(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Preprocess the dataset"""

    logger = get_run_logger() 

    try:

        X = df.drop("popularity").to_numpy()
        y = df["popularity"].to_numpy().flatten()

        # Convert popularity into 3 classes
        y_class = np.array([0 if p <= 30 else 1 if p <= 69 else 2 for p in y])

        # Log class distribution before SMOTE
        unique, counts = np.unique(y_class, return_counts=True)
        logger.info(f"📊 Class distribution before SMOTE: {dict(zip(unique, counts))}")

        # Apply SMOTE only if all 3 classes exist
        if len(unique) == 3:
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            X, y_class = smote.fit_resample(X, y_class)

        unique, counts = np.unique(y_class, return_counts=True)
        logger.info(f"📊 Class distribution after SMOTE: {dict(zip(unique, counts))}")

        # Compute class weights
        class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y_class)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        logger.info(f"⚖️ Class Weights: {class_weights_dict}")

        y_categorical = to_categorical(y_class, num_classes=3)

        return X, y_categorical, class_weights_dict

    except Exception as e:
        logger.error(f"❌ Unexpected error in load_data: {e}")
        raise

@task(name="train-split-dataset")
def split_dataset(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits dataset in train e test subdatasets."""

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=constants.TEST_SIZE, random_state=42)

    # Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, constants.SCALER_PATH)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test



@task(name="train-model")
def train_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, class_weights: dict):
    """Trains an AI model to predict music popularity."""

    logger = get_run_logger() 
    logger.info("🚀 Starting model training...")
    
    try:

        # Define the model
        model = tf.keras.Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(constants.DROPOUT_RATE_1),
            Dense(64, activation='relu'),
            Dropout(constants.DROPOUT_RATE_2),
            Dense(3, activation='softmax')
        ])

        # Compile
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])

        # Training
        logger.info("🚀 Training the model...")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=constants.EPOCHS, batch_size=constants.BATCH_SIZE, callbacks=[early_stopping], class_weight=class_weights)

        # Save model
        model.save(constants.MODEL_PATH)
        logger.info("✅ Model saved successfully!")

    except Exception as e:
        logger.error(f"❌ Error during model training: {e}")
        raise

@flow(name="train-pipeline")
def train_pipeline():
    df = load_data(constants.DATA_TRAIN_TABLENAME)
    X, y, class_weights = transform_data(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    train_model(X_train, X_test, y_train, y_test, class_weights)

if __name__ == "__main__":
    train_pipeline()
