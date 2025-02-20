from prefect import flow, task
import extract
import transform
import load
import predict
import visualize
import os

DATA_DIR = "data"
PREDICT_TRACKS_PATH = os.path.join(DATA_DIR, "new_tracks.csv")
PREDICT_TABLE_NAME = "predict_tracks"

@task
def extract_data(filename):
    return extract.extract_csv(filename)

@task
def transform_data(df):
    return transform.transform(df)

@task
def load_data(df_cleaned, table_name):
    load.load(df_cleaned, table_name)

@task
def predict_data(table_name):
    predict.predict(table_name)
    
@task
def visualize_data(table_name):
    visualize.visualize(table_name)

@flow
def music_pipeline(source_filename, predict_table_name):
    df = extract_data(source_filename)
    df_cleaned = transform_data(df)
    load_data(df_cleaned, predict_table_name)
    predict_data(predict_table_name)
    visualize_data(predict_table_name)

if __name__ == "__main__":
    source_filename = PREDICT_TRACKS_PATH
    predict_table_name = PREDICT_TABLE_NAME
    
    music_pipeline(source_filename, predict_table_name)
