FROM python:3.8.6-buster
COPY requirements.txt /requirements.txt
COPY api /api
COPY waste_classification /waste_classification
# COPY model.joblib /model.joblib
# COPY /Users/Lucy/Documents/Coding/LeWagon/lw-data-science-5ae543898fd3.json /credentials.json
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
