import polars as pl
import os
import sys

def transform(df):
    """Pulizia e trasformazione dei dati estratti."""
    print("🔄 Transforming data...")

    # Selezioniamo solo le colonne utili
    selected_columns = [
        "id",  # Identificatore univoco della traccia
        "name", # title of the track
        "artists", # artists
        "danceability",  # Indica quanto è ballabile la traccia (0-1)
        "energy",  # Misura l'intensità e l'energia della traccia (0-1)
        "tempo",  # Velocità della traccia in battiti per minuto (BPM)
        "valence",  # Misura quanto è "positiva" o "felice" la traccia (0-1)
        "loudness",  # Volume medio della traccia in decibel (dB)
        "speechiness",  # Percentuale di elementi parlati nella traccia (0-1)
        "instrumentalness",  # Probabilità che la traccia sia strumentale (0-1)
        "acousticness",  # Misura quanto è acustica la traccia (0-1)
        "mode",  # Modalità musicale della traccia (0=Minore, 1=Maggiore)
        "key",  # Tonalità della traccia (0=C, 1=C#, ..., 11=B)
        "duration_ms",  # Durata della traccia in millisecondi
        "popularity"  # Indice di popolarità della traccia (0-100)
    ]
    
    """Verifica se tutte le colonne necessarie sono presenti nel DataFrame."""
    missing_columns = [col for col in selected_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ ERRORE: Mancano le seguenti colonne nel dataset: {missing_columns}")
        sys.exit(1)  # Interrompe lo script con codice di errore
    
    # Selezioniamo solo le colonne necessarie
    df = df.select(selected_columns)

    # Rimuoviamo valori nulli
    df_cleaned = df.drop_nulls()

    # Normalizziamo le colonne numeriche tra 0 e 1 per il modello ML
    #numeric_cols = ["danceability", "energy", "tempo", "valence",
    #                "loudness", "speechiness", "instrumentalness",
    #                "acousticness", "duration_ms", "popularity"]
    
    #for col in numeric_cols:
    #    min_val = df_cleaned[col].min()
    #    max_val = df_cleaned[col].max()
    #    df_cleaned = df_cleaned.with_columns(
    #        ((df_cleaned[col] - min_val) / (max_val - min_val)).alias(col)
    #    )

    return df_cleaned

