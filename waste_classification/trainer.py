from waste_classification.data import get_data_trashnet
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomRotation, RandomFlip
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.data import AUTOTUNE
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import PIL
import PIL.Image
from glob import glob
import joblib
from google.cloud import storage
from waste_classification.data import get_data_trashnet, get_data_TACO

def augment_trashnet():
    augmentation = Sequential(
        [RandomRotation(factor=(-0.2, 0.3)),
         RandomFlip()])
    return augmentation

def normalize_trashnet():
    normalization_layer = Rescaling(1. / 255)
    return normalization_layer

def preproc_pipeline_trashnet(data_dir, model_type, save_to, epochs=5):
    train_ds, val_ds, test_ds = get_data_trashnet()
    model = train_model(train_ds, val_ds, model_type, epochs)
    model.save(save_to)

def create_main_layer(model_type, num_classes=6):
    input_shape=(180, 180, 3)
    if model_type == "ResNet50":
        from tensorflow.keras.applications import ResNet50
        base_model = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
        for layer in base_model.layers:
            layer.trainable = False
    elif model_type == "standard":
        normalization_layer = Rescaling(1./255, input_shape=input_shape)
        base_model = Sequential([
                        normalization_layer,
                        Conv2D(32, 3, activation='relu'),
                        MaxPooling2D(),
                        Conv2D(32, 3, activation='relu'),
                        MaxPooling2D(),
                        Conv2D(32, 3, activation='relu'),
                        MaxPooling2D()])
    else:
        raise Exception(f"model {model_type} not supported")
    x = tf.keras.layers.Flatten()(base_model.output)
    model = tf.keras.models.Model(base_model.input, x)
    model = Sequential([
        model,
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile()
    return model

def train_model(train_ds, val_ds, model_type, epochs):
    num_classes = 6

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    core_model = create_main_layer(model_type)
    model = Sequential([
        augment_trashnet(),
        core_model
    ])
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return core_model

def load_model(model_dir):
    return tf.keras.models.load_model(model_dir)

def save_model(reg):
    """ method that saves the model into a .joblib file and uploads it on Google Storage /models folder """
    pass

def compute_confusion_matrix(model_dir, data_dir, output_plot_fn):
    train_ds, val_ds, test_ds = load_data(data_dir)
    model = load_model(model_dir)
    confusion_matrix = None
    for batch_input, batch_output in val_ds:
        p = tf.argmax(model(batch_input), -1)
        c = tf.math.confusion_matrix(batch_output, p, num_classes=6)
        if confusion_matrix is None:
            confusion_matrix = c
        else:
            confusion_matrix += c
    labels = list(os.walk(data_dir))[0][1]
    sns.heatmap(confusion_matrix.numpy(), annot=True, xticklabels=labels, yticklabels=labels)
    plt.savefig(output_plot_fn)
    print(f"confusion matrix plot saved at {output_plot_fn}")


if __name__ == "__main__":
    model_type="standard"
    data_dir = "../../raw_data/dataset-resized/"
    model_dir = f"../../model_{model_type}"
    preproc_pipeline_trashnet(data_dir=data_dir,
                              model_type=model_type,
                              save_to=model_dir,
                              epochs=1)
    compute_confusion_matrix(model_dir, data_dir, f"../../confusion_matrix_{model_type}")
