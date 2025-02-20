from prefect import flow, task
import extract
import transform
import load
import train
import os

DATA_DIR = "data"
TRAIN_TRACKS_PATH = os.path.join(DATA_DIR, "tracks.csv")
TRAIN_TRACKS_URI = "yamaerenay/spotify-dataset-19212020-600k-tracks"
TRAIN_TABLE_NAME = "train_tracks"

@task
def extract_data(uri, filename):
    return extract.extract_kaggle(uri, filename)

@task
def transform_data(df):
    return transform.transform(df)

@task
def load_data(df_cleaned, table_name):
    load.load(df_cleaned, table_name)

@task
def train_model(table_name):
    train.train(table_name)

@flow
def music_pipeline(source_uri, source_filename, train_table_name):    
    df = extract_data(source_uri, source_filename)
    df_cleaned = transform_data(df)
    load_data(df_cleaned, train_table_name)
    train_model(train_table_name)

if __name__ == "__main__":     
    source_uri = TRAIN_TRACKS_URI
    source_filename = TRAIN_TRACKS_PATH
    train_table_name = TRAIN_TABLE_NAME
    
    music_pipeline(source_uri, source_filename, train_table_name)
