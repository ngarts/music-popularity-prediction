import polars as pl
import duckdb
import tensorflow as tf
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

DATA_DIR = "data"
MODEL_DIR = "models"
DUCKDB_PATH = os.path.join(DATA_DIR, "music_analysis.duckdb")
MODEL_PATH = os.path.join(MODEL_DIR, "nn_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Definizione delle classi di popolarità
CLASSES = {
    0: "🧊 Ice-cold (0-30)",  
    1: "🔥 Lukewarm (31-69)",  
    2: "🚀 Blazing Hot (70-100)"  
}

def load_data(table_name):
    """Carica i dati trasformati da DuckDB e prepara input e target."""
    print("📥 Loading dataset from DuckDB...")
    con = duckdb.connect(DUCKDB_PATH, read_only=False)
    
    query = f"""
        SELECT danceability, energy, tempo, valence, loudness, 
               speechiness, instrumentalness, acousticness, mode, key, duration_ms, popularity 
        FROM {table_name}
    """
    
    arrow_df = con.execute(query).fetch_arrow_table()
    df = pl.from_arrow(arrow_df)
    
    X = df.drop("popularity").to_numpy()
    y = df["popularity"].to_numpy().flatten()
    
    # Classificazione in 3 classi
    y_class = np.array([0 if p <= 30 else 1 if p <= 69 else 2 for p in y])

    # Conta il numero di campioni per classe
    unique, counts = np.unique(y_class, return_counts=True)
    print(f"📊 Distribuzione delle classi prima di SMOTE: {dict(zip(unique, counts))}")

    # Applica SMOTE solo se le classi sono sbilanciate
    if len(unique) == 3:
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X, y_class = smote.fit_resample(X, y_class)
    
    unique, counts = np.unique(y_class, return_counts=True)
    print(f"📊 Distribuzione delle classi dopo SMOTE: {dict(zip(unique, counts))}")

    # Calcoliamo i pesi delle classi
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y_class)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"⚖️ Class Weights: {class_weights_dict}")

    y_categorical = to_categorical(y_class, num_classes=3)

    return X, y_categorical, class_weights_dict

def train(table_name):
    """Allena il modello AI per predire la popolarità della musica."""
    X, y, class_weights = load_data(table_name)

    # Suddivisione train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizzazione
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, SCALER_PATH)
    
    X_test = scaler.transform(X_test)

    # Creazione del modello
    model = tf.keras.Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    
    # Compilazione
    model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
    
    # Addestramento
    print("🚀 Training the model...")
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=20, batch_size=32, callbacks=[early_stopping], class_weight=class_weights)
    
    model.save(MODEL_PATH)
    print("✅ Model saved successfully!")

if __name__ == "__main__":
    table_name = "train_tracks"
    train(table_name)
