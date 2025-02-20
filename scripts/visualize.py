import duckdb
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

DATA_DIR = "data"
DUCKDB_PATH = os.path.join(DATA_DIR, "music_analysis.duckdb")

def visualize(table_name):
    """Visualizza i dati predetti per l'analisi della popolarità delle tracce."""
    print("📊 Loading predicted data from DuckDB...")

    # Connettersi al database e caricare i dati
    with duckdb.connect(DUCKDB_PATH, read_only=False) as con:
        arrow_table = con.execute(f"SELECT name, artists, class, danceability, energy, tempo FROM {table_name}").fetch_arrow_table()
        df = pl.from_arrow(arrow_table)
        
        print(df.head())

        # 🔹 1. Grafico a barre delle classi di popolarità
        plt.figure(figsize=(8, 5))
        sns.countplot(y=df["class"], order=df["class"].value_counts().sort("class")["class"].to_list())
        plt.xlabel("Numero di Tracce")
        plt.ylabel("Classi di Popolarità")
        plt.title("Distribuzione delle Tracce per Classe di Popolarità")
        plt.show(block=False)

        # 🔹 2. Scatter plot di Danceability vs Energy, colorato per classe
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="danceability", y="energy", hue="class", palette="viridis", alpha=0.7)
        plt.xlabel("Danceability")
        plt.ylabel("Energy")
        plt.title("Danceability vs Energy per Classe di Popolarità")
        plt.show(block=False)

        # 🔹 3. Wordcloud delle tracce
        text = " ".join(df["name"])
        wordcloud = WordCloud(width=800, height=400, background_color="black").generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Wordcloud dei Titoli delle Tracce più Popolari")
        plt.show()

if __name__ == "__main__":
    table_name = "predict_tracks"
    visualize_data(table_name)
