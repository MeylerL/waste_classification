FROM python:3.8.6-buster
COPY requirements.txt /requirements.txt
COPY app.py /app.py
COPY waste_classification /waste_classification
COPY pretrained_models /pretrained_models
COPY images /images
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD streamlit run app.py
