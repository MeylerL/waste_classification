import os
package_parent = os.path.dirname(os.getcwd())
package_directory = os.getcwd()
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

if __name__ == "__main__":
  test= os.path.join(package_parent, "raw_data", "dataset-resized")
  print(test, os.path.exists(test))
