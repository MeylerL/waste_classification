
BUCKET_NAME: "wagon-data-699-waste_classification"
#RENAME WHEN WE'VE GOT THE DATAPATH
LOCAL_PATH_TRASHNET = "XXX"
LOCAL_PATH_TACO = "XXX"

BUCKET_TRAIN_DATA_PATH_TRASHNET = f"waste_management_data/{LOCAL_PATH_TRASHNET}"
BUCKET_TRAIN_DATA_PATH_TACO = f"waste_management_data/{LOCAL_PATH_TACO}"


def get_data():
    """ function used in order to get the training data (or a portion of it) from Cloud Storage """
    pass


def compute_distance(df):
    """ function used in order to compute a distance feature for a dataframe """
    pass


def preprocess(df):
    """ function that pre-processes the data """
    pass


def train_model(X_train, y_train):
    """ function that trains the model """
    pass


def save_model(reg):
    """ method that saves the model into a .joblib file and uploads it on Google Storage /models folder """
    pass


if __name__ == '__main__':
    """ runs a training """
