from re import M
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import PIL
import PIL.Image
import mlflow
import numpy as np
import time
from memoized_property import memoized_property
from glob import glob
from waste_classification.data import get_all_data, get_data_TACO, get_data_trashnet
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomRotation, RandomFlip, RandomZoom
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from google.cloud import storage
from waste_classification.params import MLFLOW_URI, LOCAL_PATH_TRASHNET, package_parent
from mlflow.tracking import MlflowClient
from tensorflow.keras.applications import DenseNet169, DenseNet121, VGG16, ResNet50, ResNet101
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from waste_classification.params import BUCKET_NAME
import argparse
from sys import argv, stderr
import sys
from tensorflow.keras.losses import Hinge

class Trainer():
    def __init__(self):
        self.experiment_name = "waste_classification_first_model"
        self.train_ds_local = None
        self.val_ds_local = None
        self.test_ds_local = None
        self.model = None
        self.mlflow = True
        self.model_type = None

    def augment_trashnet(self):
        augmentation = Sequential(
            [RandomFlip()])
        return augmentation

    def load_data(self, use_taco=False, class_balance=True, gcp=False):
        train_ds, val_ds, test_ds, class_weights = get_all_data(use_taco=use_taco,
                                                                class_balance=class_balance,
                                                                gcp=False)
        self.train_ds_local = train_ds
        self.val_ds_local = val_ds
        self.test_ds_local = test_ds
        self.class_weights = class_weights

    def create_main_layer(self, model_type="ResNet50", num_classes=6):
        model_type = "ResNet50"
        input_shape=(180, 180, 3)
        if model_type == "ResNet50":
            base_model = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
            for layer in base_model.layers:

        elif model_type == "VGG16":
            base_model = VGG16(input_shape=input_shape,
                               include_top=False,
                               weights="imagenet")
            for layer in base_model.layers:
                layer.trainable = False
        elif model_type == "DenseNet121":
            base_model = DenseNet121(include_top=False,
                                     weights="imagenet",
                                     input_shape=input_shape)
            for layer in base_model.layers:
                layer.trainable = False
        elif model_type == "DenseNet169":
            base_model = DenseNet169(include_top=False,
                                     weights="imagenet",
                                     input_shape=input_shape)
            for layer in base_model.layers:
                layer.trainable = False
        elif model_type == "standard":
            normalization_layer = Rescaling(1. / 255, input_shape=input_shape)
            base_model = Sequential([
                normalization_layer,
                Conv2D(32, 3, activation='relu'),
                MaxPooling2D(),
                Conv2D(32, 3, activation='relu'),
                MaxPooling2D(),
                Conv2D(32, 3, activation='relu'),
                MaxPooling2D()
            ])
        else:
            raise Exception(f"model {model_type} not supported")
        x = tf.keras.layers.Flatten()(base_model.output)
        model = tf.keras.models.Model(base_model.input, x)
        model = Sequential([
            model,
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')])
        model.compile()
        self.mlflow_log_param(model_type, "i am a parameter")
        return model




    def train_model(self, model_type, epochs=40):
        tic = time.time()
        core_model = self.create_main_layer(model_type)
        model = Sequential([self.augment_trashnet(), core_model])
        model.compile(Adam(),
                      loss=SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])


        has_gpu = tf.config.list_physical_devices('GPU') == []
        if has_gpu:
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
                history = model.fit(self.train_ds_local, validation_data=self.val_ds_local, epochs=epochs, callbacks=[
                    EarlyStopping(monitor='val_accuracy', patience=2), ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.0001, verbose=1)], class_weight=self.class_weights)
        else:
            history = model.fit(self.train_ds_local, validation_data=self.val_ds_local, epochs=epochs, callbacks=[
                EarlyStopping(monitor='val_accuracy', patience=2), ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.0001, verbose=1)], class_weight=self.class_weights)

        self.mlflow_log_metric("epochs", epochs)
        self.mlflow_log_metric("train_time", int(time.time() - tic))
        self.model = core_model
        self.history = history
        plt.clf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['Training', 'Validation'])
        plt.xlabel('epoch')
        plt.savefig('Accuracy.jpg')
        plt.clf()
        print(f"accuracy plot saved at Accuracy.jpg")
        val_accuracy = history.history['val_accuracy'][-1]
        print(f"model validation accuracy is {val_accuracy}")

    def load_model(self, model_location):
        print(f"loading model from {model_location}")
        self.model = tf.keras.models.load_model(model_location)

    def save_model(self, model_location):
        print(f"saving model to {model_location}")
        self.model.save(model_location)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.legend(['Training', 'Validation'])
        plt.xlabel('epoch')
        fn = "Accuracy.png"
        plt.savefig(fn)
        self.upload(fn, model_location+"_"+fn)
        self.compute_confusion_matrix(model_location+"_confusion_matrix.png")

    def compute_confusion_matrix(self, plot_location):
        plt.clf()
        model = self.model
        confusion_matrix = None
        for batch_input, batch_output in self.val_ds_local:
            p = tf.argmax(self.model(batch_input), -1)
            c = tf.math.confusion_matrix(batch_output, p, num_classes=6)
            if confusion_matrix is None:
                confusion_matrix = c
            else:
                confusion_matrix += c
        labels = ['paper', 'plastic', 'metal', 'trash', 'glass', 'cardboard']
        ax = sns.heatmap(confusion_matrix.numpy(), annot=True, xticklabels=labels, yticklabels=labels)
        ax.set(xlabel='predicted label', ylabel='true label')
        fn = "confusion_matrix.png"
        plt.savefig(fn)
        self.upload(fn, plot_location)
        plt.clf()
        print(f"confusion matrix plot saved at {plot_location}")

    def evaluate_score(self):
        model = self.model
        train_ds, val_ds, test_ds = get_data_trashnet()
        test_loss, test_acc = self.model.evaluate(test_ds)
        print('Test loss: {} Test Acc: {}'.format(test_loss, test_acc))
        self.mlflow_log_metric("Loss", test_loss)
        self.mlflow_log_metric("Accuracy", test_acc)

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def get_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.get_experiment_id)

    def mlflow_log_param(self, param_name, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id,
                                         param_name, value)

    def mlflow_log_metric(self, metric_name, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id,
                                          metric_name, value)

    def upload(self, src, tgt):
        client = storage.Client().bucket(BUCKET_NAME)

        x = tgt[tgt.index("models"):]
        blob = client.blob(x)
        blob.upload_from_filename(src)


def construct_model_location(*args):
    s = "_".join(map(str, args))
    return f"gs://{BUCKET_NAME}/models/model_{int(time.time())}_{s}"


def example_model_loading_and_prediction():
    model_location = f"gs://{BUCKET_NAME}/models/model_1631093492_ResNet50_5_True_True_True"
    model = tf.keras.models.load_model(model_location)
    batch_input = np.random.rand(4, 180, 180, 3)
    labels = ['paper', 'plastic', 'metal', 'trash', 'glass', 'cardboard']
    p = tf.argmax(model(batch_input), -1)
    p = [labels[i] for i in p]
    return p

# example_model_loading_and_prediction()
# exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", default='')
    parser.add_argument("--class-balance", default=True)
    parser.add_argument("--use-taco", default=False)
    parser.add_argument("--use-gcp", default=False)
    parser.add_argument("--model-type", default='ResNet50')
    parser.add_argument("--epochs", type=int, default=40)
    params = parser.parse_args()
    print(params)
    class_balance = params.class_balance
    use_taco = params.use_taco
    gcp = params.use_gcp
    model_type = params.model_type
    epochs = params.epochs
    model_location = construct_model_location(model_type, epochs, gcp, class_balance, use_taco)
    t = Trainer(model_type)
    t.load_data(gcp=False, class_balance=class_balance, use_taco=use_taco)
    t.train_model(model_type=model_type, epochs=epochs)
    t.save_model(model_location)

