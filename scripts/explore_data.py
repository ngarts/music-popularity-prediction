import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

TRAIN_TRACKS_PATH = os.getenv("TRAIN_TRACKS_PATH")

# Stampa un separatore per una migliore leggibilità
def print_section(title):
    print("\n" + "="*50)
    print(f"{title}")
    print("="*50)

# Carica il dataset
df = pd.read_csv(TRAIN_TRACKS_PATH)

# Mostra le prime righe
print_section("Prime righe del dataset")
print(df.head())

# Controlla le colonne disponibili
print_section("Elenco delle colonne disponibili")
print(df.columns)

# Controlliamo se ci sono valori mancanti
print_section("Valori mancanti per ogni colonna")
print(df.isnull().sum())

# Descrizione statistica delle colonne numeriche
print_section("Descrizione statistica delle colonne numeriche")
print(df.describe())

# Verifichiamo i tipi di dati
print_section("Tipi di dati delle colonne")
print(df.dtypes)
