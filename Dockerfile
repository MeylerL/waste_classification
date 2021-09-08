FROM python:3.8.6-buster
COPY requirements.txt /requirements.txt
COPY app.py /app.py
COPY pretrained_models /pretrained_models
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD streamlit run app.py --server.port 8080
