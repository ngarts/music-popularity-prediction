import duckdb
import polars as pl
import constants
from logger import logger 

def load(df: pl.DataFrame, table_name: str):
    """Loads cleaned data into a DuckDB database."""
    
    logger.info(f"📥 Loading data into DuckDB table: {table_name}...")

    try:
        # Connect to DuckDB (creates the file if it doesn't exist)
        with duckdb.connect(constants.DUCKDB_PATH, read_only=False) as con:
            
            # Drop the table if it exists to overwrite data
            con.execute(f"DROP TABLE IF EXISTS {table_name}")

            # Create table schema
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

            # Load data efficiently
            con.execute(f"INSERT INTO {table_name} SELECT * FROM df")

        logger.info(f"✅ Data successfully loaded into DuckDB table: {table_name}!")

    except duckdb.Error as e:
        logger.error(f"❌ Error loading data into DuckDB: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise
