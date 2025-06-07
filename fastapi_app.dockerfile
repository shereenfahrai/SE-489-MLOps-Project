FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fake_news_detection/ fake_news_detection/

RUN apt-get update && apt-get install -y wget

RUN mkdir -p /app/models && \
    wget https://storage.googleapis.com/mlops_fake_news/lstm_model.h5 -O /app/models/lstm_model.h5 && \
    wget https://storage.googleapis.com/mlops_fake_news/tokenizer.pkl -O /app/models/tokenizer.pkl

EXPOSE 8080
CMD ["uvicorn", "fake_news_detection.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
