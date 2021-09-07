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
data_all = data.get_all_data(gcp=False)
print("DONE")
