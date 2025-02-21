import polars as pl
import duckdb
import tensorflow as tf
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import constants
from logger import logger

# Defines popularity classes
CLASSES = {
    0: "🧊 Ice-cold (0-30)",  
    1: "🔥 Lukewarm (31-69)",  
    2: "🚀 Blazing Hot (70-100)"  
}

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 0.2
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.2

def load_data(table_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Loads and preprocesses dataset from DuckDB."""
    logger.info(f"📥 Loading dataset from DuckDB table: {table_name}...")

    try:
        con = duckdb.connect(constants.DUCKDB_PATH, read_only=True)
        query = f"""
            SELECT danceability, energy, tempo, valence, loudness, 
                   speechiness, instrumentalness, acousticness, mode, key, duration_ms, popularity 
            FROM {table_name}
        """
        
        arrow_df = con.execute(query).fetch_arrow_table()
        df = pl.from_arrow(arrow_df)
        con.close()

        if df.is_empty():
            logger.error(f"❌ No data found in table: {table_name}")
            raise ValueError("Dataset is empty!")

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

    except duckdb.Error as e:
        logger.error(f"❌ DuckDB error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in load_data: {e}")
        raise

def train_from_table(table_name: str):
    """Trains an AI model to predict music popularity."""
    logger.info("🚀 Starting model training...")
    
    try:
        X, y, class_weights = load_data(table_name)

        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

        # Normalization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        joblib.dump(scaler, constants.SCALER_PATH)
        X_test = scaler.transform(X_test)

        # Define the model
        model = tf.keras.Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(DROPOUT_RATE_1),
            Dense(64, activation='relu'),
            Dropout(DROPOUT_RATE_2),
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
                            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], class_weight=class_weights)

        # Save model
        model.save(constants.MODEL_PATH)
        logger.info("✅ Model saved successfully!")

    except Exception as e:
        logger.error(f"❌ Error during model training: {e}")
        raise

if __name__ == "__main__":
    train_from_table(constants.TRAIN_TRACKS_TABLE_NAME)
