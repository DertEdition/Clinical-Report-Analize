# Python base image
FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Requirements dosyasını kopyala ve bağımlılıkları yükle
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# Uygulama dosyalarını kopyala
COPY api.py .
COPY chat_ui.py .
COPY import_dataset.py .
COPY dataset_generator.py .
COPY buyuk_medikal_dataset.json .

# medikal_db docker-compose ile volume olarak bağlanacak

# Port
EXPOSE 8081

# Ortam değişkenleri
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://ollama:11434

# API'yi başlat
CMD ["python", "api.py"]
