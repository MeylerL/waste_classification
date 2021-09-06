#!/usr/bin/env python3

from tensorflow.data.experimental import save as save_dataset
from waste_classification import data


def save_datasets(datasets, location):
    for i, ds in enumerate(datasets):
        print(f"saving {i}")
        save_dataset(ds, f"{location}_{i}", compression="GZIP")

def load_datasets(location):
    from tensorflow.data.experimental import load as load_dataset
    return [load_dataset(f"{location}_{i}", compression="GZIP") for i in range(3)]

import pdb
print("START")
# data.save_cropped_TACO()
data = data.load_TACO(gcp=True)
# save_datasets(data, "../taco")
print("DONE")

# data = data.get_data_trashnet(gcp=False)
# save_datasets(data, "../../trashnet")


# data = data.get_data_TACO(gcp=False)
# save_datasets(data, "../../TACO")

# data2 = load_datasets("gs://wagon-data-699-waste_classification/tensorflow_datasets/trashnet")
# data2 = load_datasets("gs://wagon-data-699-waste_classification/tensorflow_datasets/TACO")
# print("SUCCESS")
# import pdb; pdb.set_trace()
