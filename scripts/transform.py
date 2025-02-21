import polars as pl
from logger import logger

def transform(df: pl.DataFrame) -> pl.DataFrame:
    """Cleans and transforms the extracted dataset."""
    
    logger.info("🔄 Transforming data...")

    # Define the required columns
    selected_columns = [
        "id",  # Unique track ID
        "name",  # Track title
        "artists",  # Artists
        "danceability",  # Danceability (0-1)
        "energy",  # Energy level (0-1)
        "tempo",  # Tempo in BPM
        "valence",  # Positivity/happiness score (0-1)
        "loudness",  # Average loudness (dB)
        "speechiness",  # Speech presence in track (0-1)
        "instrumentalness",  # Likelihood of instrumental track (0-1)
        "acousticness",  # Acoustic measure (0-1)
        "mode",  # Musical mode (0=Minor, 1=Major)
        "key",  # Key of the track (0=C, 1=C#, ..., 11=B)
        "duration_ms",  # Duration in milliseconds
        "popularity"  # Popularity index (0-100)
    ]
    
    # Check for missing columns
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"❌ ERROR: Missing columns in dataset: {missing_columns}")
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    
    # Select only necessary columns
    df = df.select(selected_columns)

    # Drop null values
    df_cleaned = df.drop_nulls()
    
    logger.info("✅ Data transformation completed successfully.")

    return df_cleaned

