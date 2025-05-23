# Base image
FROM python:3.11-slim

# Install system-level dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY pyproject.toml .
COPY Makefile .
COPY fake_news_detection/ fake_news_detection/
COPY data/ data/

# Install dependencies
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Set environment variable for module imports
ENV PYTHONPATH=.

# Default command: run the prediction script
CMD ["python", "-u", "fake_news_detection/predict_model.py"]
