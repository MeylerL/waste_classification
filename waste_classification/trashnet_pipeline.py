from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomRotationm, RandomFlip
from tensorflow.keras import Sequential

def augment_trashnet():
    augmentation = Sequential(
        [RandomRotation(factor=(-0.2, 0.3)), RandomFlip()]
    )
    return augmentation

def normalize_trashnet():
    normalization_layer = Rescaling(1./255)
    return normalization_layer

def preproc_pipeline_trashnet():
    augmentation_layer = augment_trashnet()
    normalization_layer = normalize_trashnet()
    augmentation = Sequential([
        augmentation_layer,
        normalization_layer
        #ADD MORE HERE!
        ])
