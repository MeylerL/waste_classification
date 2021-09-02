from google.cloud import storage
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path
import tempfile
from waste_classification.params import BUCKET_FOLDER, TACO_BUCKET_FILE_NAME, TRASHNET_BUCKET_FILE_NAME, BUCKET_NAME, BUCKET_TRAIN_DATA_PATH_TACO, BUCKET_TRAIN_DATA_PATH_TRASHNET
from shutil import rmtree

# output_dir = TemporaryDirectory()

dirs_to_remove = []


def cleanup_tmp_dirs():
    global dirs_to_remove
    for d in dirs_to_remove:
        rmtree(d)


def download_from_cloud(bucket_name, prefix):
    global dirs_to_remove
    output_dir = tempfile.mkstemp()[1]
    dirs_to_remove.append(output_dir)
    os.remove(output_dir)
    print(f"downloading data from Google cloud to {output_dir}")

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_or_name=bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        print(f"downloading {blob.name}")
        file_split = blob.name.split("/")
        directory = "/".join([output_dir]+file_split[0:-1])
        fn = os.path.join(directory, file_split[-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(fn)
    print("downloading data from Google cloud DONE")
    return output_dir


def load_trashnet(gcp=False):
    """loads trashnet data from files within gcp if gcp True. Otherwise loads from local dataset.
    Returns train_ds, val_ds, test_ds as pandas dataframes."""
    if gcp:
      # directory = f"gs://{BUCKET_NAME}/{BUCKET_FOLDER}/{TRASHNET_BUCKET_FILE_NAME}"
      directory = download_from_cloud(bucket_name=BUCKET_NAME, prefix=f"{BUCKET_FOLDER}/{TRASHNET_BUCKET_FILE_NAME}")
    else:
      #uses resized dataset on local computer
      directory = "../../raw_data/dataset-resized/"
    batch_size = 32
    img_height = 180
    img_width = 180
    all_train_ds = image_dataset_from_directory(directory,
                                                validation_split=0.2,
                                                subset="training",
                                                seed=123,
                                                image_size=(
                                                    img_height, img_width),
                                                batch_size=batch_size)
    valid_batches = len(all_train_ds)//5
    train_ds = all_train_ds.skip(valid_batches)
    val_ds = all_train_ds.take(valid_batches)
    test_ds = image_dataset_from_directory(directory,
                                          validation_split=0.2,
                                          subset="validation",
                                          seed=123,
                                          image_size=(img_height, img_width),
                                          batch_size=batch_size)
    return train_ds, val_ds, test_ds


def load_TACO(gcp=False):
    """loads TACOS data from files within gcp if gcp True. Otherwise loads from local dataset.
    Returns train_ds, val_ds, test_ds as pandas dataframes."""
    if gcp:
      # directory = f"gs://{BUCKET_NAME}/{BUCKET_FOLDER}/{TACO_BUCKET_FILE_NAME}"
      directory = download_from_cloud(bucket_name=BUCKET_NAME, prefix=f"{BUCKET_FOLDER}/{TACO_BUCKET_FILE_NAME}")
    else:
      #uses dataset on local computer. For this, make sure that the folders called
      #paper, plastic, trash, cardboard, metal are in a folder called "cat_fodlers" under TACO/data
      directory = "../raw_data/TACO/data/cat_folders/"
    batch_size = 32
    img_height = 180
    img_width = 180
    all_train_ds = image_dataset_from_directory(directory,
                                                validation_split=0.2,
                                                subset="training",
                                                seed=123,
                                                image_size=(
                                                    img_height, img_width),
                                                batch_size=batch_size)
    valid_batches = len(all_train_ds)//5
    train_ds = all_train_ds.skip(valid_batches)
    val_ds = all_train_ds.take(valid_batches)
    test_ds = image_dataset_from_directory(directory,
                                          validation_split=0.2,
                                          subset="validation",
                                          seed=123,
                                          image_size=(img_height, img_width),
                                          batch_size=batch_size)
    return train_ds, val_ds, test_ds

def get_data_TACO():
    '''returns a TACO train_ds, val_ds, test_ds from GCP'''
    train_ds, val_ds, test_ds = load_TACO(gcp=True)
    return train_ds, val_ds, test_ds

def get_data_trashnet():
    '''returns a trashnet train_ds, val_ds, test_ds from GCP'''
    print("hi")
    train_ds, val_ds, test_ds = load_trashnet(gcp=True)
    return train_ds, val_ds, test_ds

if __name__ == '__main__':
    df, df1, df2 = get_data_trashnet()
    print(len(df))
    cleanup_tmp_dirs()
    print("SUCCESS")
