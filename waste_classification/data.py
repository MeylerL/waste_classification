from google.cloud import storage
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras.preprocessing import image_dataset_from_directory
from waste_classification.params import BUCKET_FOLDER, TACO_BUCKET_FILE_NAME, TRASHNET_BUCKET_FILE_NAME, BUCKET_NAME, BUCKET_TRAIN_DATA_PATH_TACO, BUCKET_TRAIN_DATA_PATH_TRASHNET
from waste_classification.params import TACO_path, annotations_path
import json
import numpy as np
import os.path
from PIL import Image, ImageFilter

def load_trashnet(gcp=False):
    """loads trashnet data from files within gcp if gcp True. Otherwise loads from local dataset.
    Returns train_ds, val_ds, test_ds as pandas dataframes."""
    if gcp:
      directory = f"gs://{BUCKET_NAME}/{BUCKET_FOLDER}/{TRASHNET_BUCKET_FILE_NAME}"
    else:
      #uses resized dataset on local computer
      directory = "../raw_data/dataset-resized/"
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
      directory = f"gs://{BUCKET_NAME}/{BUCKET_FOLDER}/{TACO_BUCKET_FILE_NAME}"
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

def save_cropped_TACO():
  with open(annotations_path, 'r') as f:
    dataset = json.loads(f.read())
    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_annotations = len(anns)
    category_conversion = {}
    category_conversion['metal'] = [0, 8, 10, 11, 12, 28]
    category_conversion['cardboard'] = [13, 14, 15, 16, 17, 18, 19, 20]
    category_conversion['glass'] = [6, 9, 23, 26]
    category_conversion['paper'] = [21, 30, 31, 32, 33, 34]
    category_conversion['plastic'] = [4, 5, 7, 24, 27, 43, 44, 47, 49, 55]
    category_conversion['trash'] = [1, 2, 3, 22, 25, 29, 35, 36, 37,
                                38, 39, 40, 41, 42, 45, 46, 48, 50, 51, 52, 53, 54, 57, 58, 59]
    cropping_df = pd.DataFrame(columns=["image_id", "cat_name"])
    for ann in range(nr_annotations):
        for cat_name, type_nums in category_conversion.items():
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
    df.to_csv(TACO_path + 'InitialData.csv', index=False)
    # path of the folder containing the original images
    inPath = TACO_path+'data'
    # path of the folder that will contain the cropped image
    #must create trainDataTACO folder locally! It will rest inside  TACO.
    #must also create subfolders inside trainDataTACO with names of all cetegories
    outPath = TACO_path+'trainDataTACO'
    df.reset_index(inplace=True, drop=True)
    # Save cropped images in a new directory
    for ind in df.index:
        bbox = (df['x_min'][ind], df['y_min'][ind],
                df['x_max'][ind], df['y_max'][ind])
        imagePath = os.path.join(inPath+'/'+df['filename'][ind])
        img = Image.open(imagePath)
        img = img.crop(bbox)
        imageName = df["filename"][ind].split(
            '/')[0]+df["filename"][ind].split('/')[1]
        imageName = imageName[:-4]
        folder_name = df["category"][ind]
        croppedImagePath = outPath + '/' + folder_name + \
            "/" + imageName + 'cropped'+'.jpg'
        img.save(croppedImagePath)

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
    save_cropped_TACO()
