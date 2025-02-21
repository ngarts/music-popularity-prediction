import duckdb
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import constants
from logger import logger

def visualize_data(table_name: str):
    """Visualizes predicted data for track popularity analysis."""
    
    logger.info("📊 Loading predicted data from DuckDB...")

    try:
        # Connect to the database and load data
        with duckdb.connect(constants.DUCKDB_PATH, read_only=True) as con:
            query = f"""
                SELECT name, artists, class, danceability, energy, tempo 
                FROM {table_name}
            """
            arrow_table = con.execute(query).fetch_arrow_table()
            df = pl.from_arrow(arrow_table)

            # Check if the table is empty
            if df.is_empty():
                logger.error("❌ No data found in the table.")
                raise ValueError("The table is empty, no data to visualize.")

            logger.info("✅ Data successfully loaded.")
            logger.info(f"\n{df.head()}")  # Log the first few rows

            # 🔹 1. Bar chart of popularity class distribution
            plt.figure(figsize=(8, 5))
            sns.countplot(y=df["class"], order=df["class"].value_counts().sort("class")["class"].to_list())
            plt.xlabel("Number of Tracks")
            plt.ylabel("Popularity Class")
            plt.title("Distribution of Tracks by Popularity Class")
            plt.tight_layout()
            plt.show(block=False)

            # 🔹 2. Scatter plot of Danceability vs Energy, colored by class
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x="danceability", y="energy", hue="class", palette="viridis", alpha=0.7)
            plt.xlabel("Danceability")
            plt.ylabel("Energy")
            plt.title("Danceability vs Energy by Popularity Class")
            plt.tight_layout()
            plt.show(block=False)

            # 🔹 3. WordCloud of track names
            text = " ".join(df["name"].drop_nulls().to_list())  # Ensure no NaN values

            if text.strip():  # Check if there's text to display
                wordcloud = WordCloud(width=800, height=400, background_color="black").generate(text)

                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title("WordCloud of Most Popular Track Titles")
                plt.tight_layout()
                plt.show()
            else:
                logger.warning("⚠️ No valid track names found for WordCloud visualization.")

    except duckdb.Error as e:
        logger.error(f"❌ DuckDB error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in visualize_data: {e}")
        raise

if __name__ == "__main__":
    visualize_data(constants.PREDICT_TABLE_NAME)
