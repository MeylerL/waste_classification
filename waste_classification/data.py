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
from waste_classification.params import BUCKET_FOLDER, TACO_BUCKET_PREFIX, TRASHNET_BUCKET_PREFIX, BUCKET_NAME, TRASHNET_RESIZED, LOCAL_PATH_TACO
from PIL import Image, ImageFilter
from shutil import rmtree
from waste_classification.params import TACO_path, annotations_path, CATEGORY_CONVERSION
import json
import os.path
from PIL import Image, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_trashnet(gcp=False):
    if gcp:
        directory = f"gs://{BUCKET_NAME}/{TRASHNET_BUCKET_PREFIX}"
        from tensorflow.data.experimental import load as load_dataset
        return tuple(load_dataset(f"{directory}_{i}", compression="GZIP") for i in range(3))
    else:
        directory = TRASHNET_RESIZED
        batch_size = 32
        img_height = 180
        img_width = 180
        all_train_ds = image_dataset_from_directory(directory,
                                                    validation_split=0.2,
                                                    subset="training",
                                                    seed=123,
                                                    image_size=(img_height, img_width),
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


def delete_trained_TACO_folders():
    folder = LOCAL_PATH_TACO
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            rmtree(file_path)


def create_trained_TACO_folders():
    categories =["paper", "plastic", "trash", "cardboard", "metal", "glass"]
    parent_dir = LOCAL_PATH_TACO
    for category in categories:
        path = os.path.join(parent_dir, category)
        os.makedirs(path)
        print("Directory '% s' created" % category)


def get_data_TACO(gcp=False):
    """loads TACOS data from files within gcp if gcp True. Otherwise loads from local dataset."""
    if gcp:
        directory = f"gs://{BUCKET_NAME}/{TACO_BUCKET_PREFIX}"
        from tensorflow.data.experimental import load as load_dataset
        return tuple(load_dataset(f"{directory}_{i}", compression="GZIP") for i in range(3))
    else:
        directory = LOCAL_PATH_TACO
        batch_size = 16
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
        if ind % 100 == 0:
            print(f"processed {ind}")
        bbox = (df['x_min'][ind], df['y_min'][ind],
                df['x_max'][ind], df['y_max'][ind])
        imagePath = os.path.join(inPath, df['filename'][ind])
        img = Image.open(imagePath)
        img = img.crop(bbox)
        imageName = df["filename"][ind].split(
            '/')[0]+df["filename"][ind].split('/')[1]
        imageName = imageName[:-4]
        folder_name = df["category"][ind]
        Path(os.path.join(outPath, folder_name)).mkdir(parents=True, exist_ok=True)
        croppedImagePath = os.path.join(
            outPath, folder_name,  imageName+'cropped.jpg')
        img.save(croppedImagePath)


def get_all_data(gcp=False, class_balance=True, use_taco=False):
    labels = ['paper', 'plastic', 'metal', 'trash', 'glass', 'cardboard']
    trashnet_class_count = np.array([594, 482, 410, 137, 501, 403])
    TACO_class_count = np.array([260, 1004, 276, 404, 95, 46])

    class_count = trashnet_class_count
    train, val, test = get_data_trashnet(gcp)
    if use_taco:
        class_count += TACO_class_count
        for x in get_data_TACO(gcp):
            train = train.concatenate(x)
    if class_balance:
        class_weights = np.ones(6)
    else:
        class_weights = 6*class_count / class_count.sum()
    return train, val, test, dict(enumerate(class_weights))


if __name__ == '__main__':
    delete_trained_TACO_folders()
    print("FOLDERS DELETED!!")
    create_trained_TACO_folders()
    save_cropped_TACO()
