import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import PIL
import PIL.Image
import mlflow
import numpy as np
from glob import glob
from waste_classification.data import get_data_trashnet
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomRotation, RandomFlip
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from google.cloud import storage
from waste_classification.data import get_data_trashnet
from mlflow.tracking import MlflowClient


class Trainer():
    def __init__(self):
        self.experiment_name = "waste_classification_first_model"
        self.train_ds_local = None
        self.val_ds_local = None
        self.test_ds_local = None

    def augment_trashnet(self):
        augmentation = Sequential(
            [RandomRotation(factor=(-0.2, 0.3)),
                RandomFlip()])
        return augmentation


    def preproc_pipeline_trashnet(self, data_dir, model_type, save_to, epochs=5):
        train_ds, val_ds, test_ds = get_data_trashnet()
        self.train_ds_local = train_ds
        self.val_ds_local = val_ds
        self.test_ds_local = test_ds
        # model = Trainer.train_model(train_ds, val_ds, model_type, epochs)
        # model = Trainer.train_model(self.train_ds_local, self.val_ds_local, model_type, epochs)
        # model.save(save_to)


    def create_main_layer(model_type="standard", num_classes=6):
        model_type = "ResNet50"
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

    def train_model(self, model_type, epochs=1):
        # AUTOTUNE = tf.data.experimental.AUTOTUNE
        # train_ds = self.train_ds_local.cache().prefetch(buffer_size=AUTOTUNE)
        # val_ds = self.val_ds_local.cache().prefetch(buffer_size=AUTOTUNE)
        core_model = self.create_main_layer()
        model = Sequential([
            self.augment_trashnet(),
            core_model
        ])
        model.compile(optimizer='adam',
                        loss=SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['accuracy'])
        # model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        model.fit(self.train_ds_local, validation_data=self.val_ds_local, epochs=epochs)
        return model

    def load_model(self, model_dir):
        return tf.keras.models.load_model(model_dir)

    # def
    # """ method that saves the model into a .joblib file and uploads it on Google Storage /models folder """
    # pass

    def compute_confusion_matrix(self, model_dir, data_dir, output_plot_fn):
        train_ds, val_ds, test_ds = get_data_trashnet()
        model = self.load_model(model_dir)
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

    def mlflow_client(self):
        client = MlflowClient()
        mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
        return client

    def get_experiment_id(self):
        try:
            return self.mlflow_client().create_experiment(self.experiment_name)
        except:
            return self.mlflow_client().get_experiment_by_name(self.experiment_name).experiment_id

    def mlflow_run(self):
        return self.mlflow_client().create_run(self.get_experiment_id())

    def mlflow_log_param(self, param_name, value):
        self.mlflow_client().log_param(self.mlflow_run().info.run_id, param_name, value)

    def mlflow_log_metric(self, metric_name, value):
        self.mlflow_client().log_metric(self.mlflow_run().info.run_id, metric_name, value)



if __name__ == "__main__":
    data_dir = "../../raw_data/dataset-original/"
    model_dir = "../../model_standard"
    t = Trainer()
    t.preproc_pipeline_trashnet(data_dir=data_dir,
                              model_type="standard",
                              save_to=model_dir,
                              epochs=1)
    t.train_model(model_type="standard")

    # t.compute_confusion_matrix(model_dir, data_dir, f"../../confusion_matrix_{model_type}")
