import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
from tensorflow.keras.preprocessing import image_dataset_from_directory


def load_trashnet(colab=False):
  "loads trashnet data from files within raw_data folder.Returns a train_ds and val_ds as pandas dataframes."
  if colab:
    #uses original dataset since on virtual machine
    directory = "/content/dataset-original"
  else:
    #uses resized dataset on local computer
    directory = "../raw_data/dataset-resized/"
  batch_size = 32
  img_height = 180
  img_width = 180
  train_ds = image_dataset_from_directory(
      directory,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
  val_ds = image_dataset_from_directory(
      directory,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
  return train_ds, val_ds


def load_tacos(colab=False):
  "loads TACO data from files within raw_data/TACOS/data folder.Returns a train_ds and val_ds as pandas dataframes."
  if colab:
    #uses original dataset since on virtual machine
    #paper, plastic, trash, cardboard, metal are in a folder called "cat_fodlers" under /content.
    directory = "/content/cat_folders"
  else:
    #uses dataset on local computer. For this, make sure that the folders called
    #paper, plastic, trash, cardboard, metal are in a folder called "cat_fodlers" under TACO/data
    directory = "../raw_data/TACO/data/cat_folders/"
  batch_size = 32
  img_height = 180
  img_width = 180
  train_ds = image_dataset_from_directory(
      directory,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

  val_ds = image_dataset_from_directory(
      directory,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
  return train_ds, val_ds
