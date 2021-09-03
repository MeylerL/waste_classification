import os
package_parent = os.path.dirname(os.getcwd())
package_directory = os.getcwd()

# for trainer
MLFLOW_URI = "https://mlflow.lewagon.co/"

#general paths:
TRASHNET_RESIZED = os.path.join(package_parent, "raw_data", "dataset-resized")
#for GCP
BUCKET_NAME = "wagon-data-699-waste_classification"
LOCAL_PATH_TRASHNET = os.path.join(package_parent, "raw_data", "dataset-original")
LOCAL_PATH_TACO = os.path.join(package_parent, "raw_data", "TACO", "trainDataTACO")
BUCKET_TRAIN_DATA_PATH_TRASHNET = f"waste_management_data/{LOCAL_PATH_TRASHNET}"
BUCKET_TRAIN_DATA_PATH_TACO = f"waste_management_data/{LOCAL_PATH_TACO}"
BUCKET_FOLDER = "waste_management_data"
TRASHNET_BUCKET_FILE_NAME = "dataset-original"
TACO_BUCKET_FILE_NAME = "cat_folder"

#for taco cropping:
TACO_path = os.path.join(package_parent, "raw_data", "TACO")
annotations_path = os.path.join(package_parent, "raw_data", "TACO", "data", "annotations.json")

#category assignment:
CATEGORY_CONVERSION = {}
CATEGORY_CONVERSION['metal'] = [0, 8, 10, 11, 12, 28, 52]
CATEGORY_CONVERSION['cardboard'] = [13, 17, 19]
CATEGORY_CONVERSION['glass'] = [6, 9, 23, 26]
CATEGORY_CONVERSION['paper'] = [21, 30, 31, 32, 33, 34, 20]
CATEGORY_CONVERSION['plastic'] = [
    4, 5, 7, 24, 27, 44, 47, 49, 55, 29, 21, 36, 38, 39]
CATEGORY_CONVERSION['trash'] = [1, 3, 22, 35, 42, 46, 51, 57, 58]
