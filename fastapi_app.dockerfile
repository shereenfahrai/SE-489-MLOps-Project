FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fake_news_detection/ fake_news_detection/

RUN apt-get update && apt-get install -y wget

RUN mkdir -p /app/models && \
    wget https://github.com/LiangcaiXie/SE-489-MLOps-Project/releases/download/v1.0.0/lstm_model.h5 -O /app/models/lstm_model.h5 && \
    wget https://github.com/LiangcaiXie/SE-489-MLOps-Project/releases/download/v1.0.0/tokenizer.pkl -O /app/models/tokenizer.pkl


EXPOSE 8080
CMD ["uvicorn", "fake_news_detection.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
