import duckdb
import polars as pl
import os

DATA_DIR = "data"
DUCKDB_PATH = os.path.join(DATA_DIR, "music_analysis.duckdb")

def load(df_cleaned, table_name):
    """Carica i dati puliti in DuckDB."""
    print(f"Loading data into DuckDB table: {table_name}...")
    
    # Connettersi al database DuckDB (crea il file se non esiste)
    con = duckdb.connect(DUCKDB_PATH, read_only=False)
    
    # Eliminare la tabella se esiste già (così ogni volta i dati vengono riscritti)
    drop_table_query = f"DROP TABLE IF EXISTS {table_name}"
    con.execute(drop_table_query)

    # Creare la tabella se non esiste
    con.execute(f"""
        CREATE TABLE {table_name} (
            id STRING,
            name STRING,
            artists STRING,
            danceability FLOAT,
            energy FLOAT,
            tempo FLOAT,
            valence FLOAT,
            loudness FLOAT,
            speechiness FLOAT,
            instrumentalness FLOAT,
            acousticness FLOAT,
            mode INTEGER,
            key INTEGER,
            duration_ms FLOAT,
            popularity FLOAT
        )
    """)
    
    # Caricare i dati in batch
    batch_size = 10000  # Imposta un batch più piccolo se il dataset è grande
    for i in range(0, len(df_cleaned), batch_size):
        batch = df_cleaned[i:i + batch_size]
        con.execute(f"INSERT INTO {table_name} SELECT * FROM batch")
    
    print("✅ Data successfully loaded into DuckDB!")

