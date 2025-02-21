import duckdb
import constants

def check_database(table_name):
    """Verifica il contenuto del database DuckDB"""
    with duckdb.connect(constants.DUCKDB_PATH, read_only=False) as con:

        # Elenca tutte le tabelle nel database
        tables = con.execute("SELECT DISTINCT * FROM predict_tracks LIMIT 10").fetch_arrow_table()
        print("📂 Tabelle presenti nel database:")
        print(tables)

        # Scegli una tabella e controlla il numero di righe
        #row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        #print(f"🔍 Numero totale di righe in {table_name}: {row_count}")

        # Visualizza alcune righe di esempio
        #sample_data = con.execute(f"SELECT * FROM {table_name} LIMIT 10").fetchdf()
        #print("🔎 Anteprima dati:")
        #print(sample_data)


if __name__ == "__main__":
    check_database(constants.TRAIN_TABLE_NAME)
    