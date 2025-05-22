# Base image
FROM python:3.11-slim

# Install system dependencies
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

# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Set PYTHONPATH so modules resolve correctly
ENV PYTHONPATH=.

# Run training script
CMD ["python", "-u", "fake_news_detection/train_model.py"]
