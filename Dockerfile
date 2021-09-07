FROM python:3.8.6-buster
COPY requirements.txt /requirements.txt
COPY app.py /app.py
COPY pretrained_model /pretrained_model
