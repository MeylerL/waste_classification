from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return "Hello from Cloud Run CD"

@app.get("/predict")
def predict():
    #### CODE HERE !!! ####
