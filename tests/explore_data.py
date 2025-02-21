import pandas as pd
import constants

# Stampa un separatore per una migliore leggibilità
def print_section(title):
    print("\n" + "="*50)
    print(f"{title}")
    print("="*50)

# Carica il dataset
df = pd.read_csv(constants.TRAIN_TRACKS_FILENAME)

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
