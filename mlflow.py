import mlflow
from mlflow.tracking import MlflowClient
from tensorflow.keras.metrics import SparseCategoricalCrossentropy
from memoized_property import memoized_property

class Trainer():
    ## move to real trainer class in future!
    def __init__(self):
        self.experiment_name = "waste_classification"

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

    def train(self, models={"sample_model": "model_object"}, metrics={"sample_metric": 0}):
        """Must pass a dictionary of model names as keys and objects as values and
        a dictionary of metrics with metric names as keys and metric values as values"""
        for model_name, model_value in models.items():
            self.mlflow_run()
            self.mlflow_log_param(model_name, model_value)
            for metric_name, metric_value in metrics.items():
                self.mlflow_log_metric(metric_name, metric_value)

#to test:
# trainer = Trainer()
# trainer.train()
