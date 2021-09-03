from google.cloud import storage
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path
import tempfile
from waste_classification.params import BUCKET_FOLDER, TACO_BUCKET_FILE_NAME, TRASHNET_BUCKET_FILE_NAME, BUCKET_NAME, TRASHNET_RESIZED, LOCAL_PATH_TACO
from PIL import Image, ImageFilter
from shutil import rmtree
from waste_classification.params import TACO_path, annotations_path, CATEGORY_CONVERSION
import json
import numpy as np
import os.path
from PIL import Image, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
        directory = TRASHNET_RESIZED
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


def load_trashnet_datagen(gcp=False):
    """loads trashnet data from files within gcp if gcp True. Otherwise loads from local dataset.
    Returns train_ds, val_ds, test_ds as pandas dataframes."""
    if gcp:
        # directory = f"gs://{BUCKET_NAME}/{BUCKET_FOLDER}/{TRASHNET_BUCKET_FILE_NAME}"
        directory = download_from_cloud(bucket_name=BUCKET_NAME, prefix=f"{BUCKET_FOLDER}/{TRASHNET_BUCKET_FILE_NAME}")
    else:
        #uses resized dataset on local computer
        directory = TRASHNET_RESIZED
    batch_size = 32
    img_height = 180
    img_width = 180
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5,
                                       horizontal_flip=True,
                                       zoom_range=0.4,
                                       shear_range=45.0,
                                       validation_split=0.2,
                                       dtype='float32')

    valid_datagen = ImageDataGenerator(rescale=1/ 255.0,
                                       validation_split=0.2,
                                       dtype='float32')

    test_datagen  = ImageDataGenerator(rescale = 1.0/ 255.0)

    train_ds = train_datagen.flow_from_directory(directory=directory,
                                                 target_size=(img_height,
                                                              img_width),
                                                 class_mode='categorical',
                                                 batch_size=batch_size,
                                                 subset='training')
    valid_ds = valid_datagen.flow_from_directory(directory=directory,
                                                 target_size=(img_height,
                                                              img_width),
                                                 class_mode='categorical',
                                                 batch_size=batch_size,
                                                 subset='validation')

    test_ds = test_datagen.flow_from_directory(directory=directory,
                                               target_size=(img_height,
                                                            img_width),
                                               class_mode='categorical',
                                               batch_size=batch_size,
                                               shuffle=False)
    return train_ds, valid_ds, test_ds



def load_TACO(gcp=False):
    """loads TACOS data from files within gcp if gcp True. Otherwise loads from local dataset.
    Returns train_ds, val_ds, test_ds as pandas dataframes."""
    if gcp:
        # directory = f"gs://{BUCKET_NAME}/{BUCKET_FOLDER}/{TACO_BUCKET_FILE_NAME}"
        directory = download_from_cloud(bucket_name=BUCKET_NAME, prefix=f"{BUCKET_FOLDER}/{TACO_BUCKET_FILE_NAME}")
    else:
        #uses dataset on local computer. For this, make sure that the folders called
        #paper, plastic, trash, cardboard, metal are in a folder called "cat_fodlers" under TACO/data
        directory = LOCAL_PATH_TACO
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

def save_cropped_TACO():
    with open(annotations_path, 'r') as f:
        dataset = json.loads(f.read())
        categories = dataset['categories']
        anns = dataset['annotations']
        imgs = dataset['images']
        nr_annotations = len(anns)
        cropping_df = pd.DataFrame(columns=["image_id", "cat_name"])
        for ann in range(nr_annotations):
            for cat_name, type_nums in CATEGORY_CONVERSION.items():
                if anns[ann]["category_id"] in type_nums:
                    cropping_df = cropping_df.append({"image_id": anns[ann]["image_id"], "segmentation": anns[ann]["segmentation"], "area": anns[ann]
                                                    ["area"], "iscrowd": anns[ann]["iscrowd"], "bbox": anns[ann]["bbox"], "cat_name": cat_name}, ignore_index=True)
        file_name = []
        for image_id in cropping_df["image_id"]:
            for img in imgs:
                if img["id"] == image_id:
                    file_name.append(img["file_name"])
        file_name = pd.Series(file_name)
        cropping_df["file_name"] = file_name
        df = pd.concat([cropping_df, cropping_df['bbox'].apply(pd.Series)], axis=1)
        df.columns = ['image_id', 'category', 'area', 'bbox', 'iscrowd', 'segmentation',
                      'filename', 'x_min', 'y_min', 'x_max', 'y_max']  # x_max : width and y_max : height
        df = df.drop(['bbox', "image_id", "area", "iscrowd", "segmentation"], axis=1)
        # Calculate maximum x and maximum y points
        df['x_max'] = df['x_max']+df['x_min']
        df['y_max'] = df['y_max']+df['y_min']
        # Convert float columns to integer
        for col in df.columns[2:]:
            df[col] = df[col].astype(int)
        #Add padding to the bounding boxes
        padding = 20
        df['x_min'] = df['x_min'] - padding
        df['y_min'] = df['y_min'] - padding
        df['x_max'] = df['x_max'] + padding
        df['y_max'] = df['y_max'] + padding
        df.to_csv(os.path.join(TACO_path, 'InitialData.csv'), index=False)
        # path of the folder containing the original images
        inPath = os.path.join(TACO_path, 'data')
        # path of the folder that will contain the cropped image
        #must create trainDataTACO folder locally! It will rest inside  TACO.
        #must also create subfolders inside trainDataTACO with names of all cetegories
        outPath = os.path.join(TACO_path, 'trainDataTACO')
        df.reset_index(inplace=True, drop=True)
        # Save cropped images in a new directory
        for ind in df.index:
            bbox = (df['x_min'][ind], df['y_min'][ind],
                    df['x_max'][ind], df['y_max'][ind])
            imagePath = os.path.join(inPath, df['filename'][ind])
            img = Image.open(imagePath)
            img = img.crop(bbox)
            imageName = df["filename"][ind].split(
                '/')[0]+df["filename"][ind].split('/')[1]
            imageName = imageName[:-4]
            folder_name = df["category"][ind]
            croppedImagePath = os.path.join(outPath, folder_name, imageName,'cropped.jpg')
            img.save(croppedImagePath)

def get_data_TACO():
    '''returns a TACO train_ds, val_ds, test_ds from GCP'''
    train_ds, val_ds, test_ds = load_TACO(gcp=False)
    return train_ds, val_ds, test_ds

def get_data_trashnet():
    '''returns a trashnet train_ds, val_ds, test_ds from GCP'''
    print("hi")
    train_ds, val_ds, test_ds = load_trashnet_datagen(gcp=False)
    return train_ds, val_ds, test_ds

if __name__ == '__main__':
    df, df1, df2 = get_data_trashnet()
    print(len(df))
    cleanup_tmp_dirs()
    print("SUCCESS")
