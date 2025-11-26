FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:savoury1/ffmpeg4 \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории и структуры папок для всех моделей
WORKDIR /app
RUN mkdir -p /app/models/whisper /app/models/speechbrain /app/models/qwen /app/logs /app/uploads /app/results

# Копирование requirements
COPY requirements.txt .

# Обновление pip и установка зависимостей
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Установка дополнительных пакетов для мониторинга памяти
RUN pip3 install pynvml psutil

# Скачивание и сохранение моделей локально при сборке
# 1. Whisper модель
RUN python3 -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
WhisperProcessor.from_pretrained('bond005/whisper-podlodka-turbo'); \
WhisperForConditionalGeneration.from_pretrained('bond005/whisper-podlodka-turbo'); \
print('Whisper model downloaded')"

# 2. SpeechBrain модель
RUN python3 -c "from speechbrain.pretrained import EncoderClassifier; \
EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='/app/models/speechbrain'); \
print('SpeechBrain model downloaded')"

# 3. Qwen модель без квантования (скачивается при сборке)
RUN pip3 install huggingface_hub
RUN python3 -c "from huggingface_hub import snapshot_download; \
import os; \
snapshot_download(repo_id='Qwen/Qwen3-4B-Instruct-2507', local_dir='/app/models/qwen', local_dir_use_symlinks=False); \
print('Qwen model downloaded successfully')"

# Проверка скачанных моделей
RUN echo "=== Проверка скачанных моделей ===" && \
    echo "Whisper:" && find /app/models/whisper/ -name "*.json" | head -5 && \
    echo "SpeechBrain:" && find /app/models/speechbrain/ -name "*.json" | head -5 && \
    echo "Qwen:" && find /app/models/qwen/ -name "*.json" | head -5 && \
    echo "Config file check:" && [ -f /app/models/qwen/config.json ] && echo "✅ Qwen config.json exists" || echo "❌ Qwen config.json missing"

# Копирование исходного кода
COPY task_manager.py /app/
COPY main.py /app/
COPY processing/ /app/processing/
COPY static/ /app/static/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
