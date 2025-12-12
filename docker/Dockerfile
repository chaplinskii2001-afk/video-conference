FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Аргументы сборки
ARG HF_TOKEN
ARG EDUCATIONAL_MODE=true
ARG PERFORMANCE_LEVEL=balanced

# Переменные окружения для образовательной среды
ENV EDUCATIONAL_MODE=${EDUCATIONAL_MODE}
ENV PERFORMANCE_LEVEL=${PERFORMANCE_LEVEL}
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Установка системных зависимостей
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    ubuntu-keyring \
    gnupg \
    python3.10 \
    python3-pip \
    libpython3.10 \
    ffmpeg \
    git \
    git-lfs \
    wget \
    curl \
    libsndfile1 \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Создание рабочей директории и структуры папок
WORKDIR /app
RUN mkdir -p /app/models/whisper /app/models/pyannote /app/models/qwen /app/logs /app/uploads /app/results /app/data

# Установка Python зависимостей
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install "pyannote.audio @ git+https://github.com/pyannote/pyannote-audio.git" \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --no-cache-dir

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .

# Дополнительные пакеты для образовательной среды
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir \
    pynvml psutil huggingface_hub \
    jupyter \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    rich \
    typer \
    pydantic[email] \
    fastapi[all] \
    uvicorn[standard] \
    gpustat

# Загрузка AI моделей с обработкой ошибок для образовательной среды
RUN echo "=== Загрузка AI моделей для образовательной среды ===" && \
    echo "Начинаем загрузку Whisper модели..." && \
    python3 -c "
import os
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    WhisperProcessor.from_pretrained('bond005/whisper-podlodka-turbo')
    WhisperForConditionalGeneration.from_pretrained('bond005/whisper-podlodka-turbo')
    print('✅ Whisper model loaded successfully')
except Exception as e:
    print(f'❌ Whisper model loading failed: {e}')
    print('⚠️  Whisper will be downloaded at runtime')
" || echo "⚠️  Whisper будет загружен при первом запуске"

RUN echo "Загружаем PyAnnote модель..." && \
    python3 -c "
import os
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    try:
        from pyannote.audio import Pipeline
        Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token=hf_token, cache_dir='/app/models/pyannote')
        print('✅ PyAnnote model loaded successfully')
    except Exception as e:
        print(f'❌ PyAnnote model loading failed: {e}')
        print('⚠️  PyAnnote will be downloaded at runtime')
else:
    print('⚠️  HF_TOKEN not provided, PyAnnote will be downloaded at runtime')
" || echo "⚠️  PyAnnote будет загружен при первом запуске"

RUN echo "Загружаем Qwen модель..." && \
    python3 -c "
try:
    from huggingface_hub import snapshot_download
    import os
    # Проверяем наличие модели локально
    if not os.path.exists('/app/models/qwen/config.json'):
        print('📥 Downloading Qwen model for educational use...')
        snapshot_download(repo_id='Qwen/Qwen3-4B-Instruct-2507', 
                         local_dir='/app/models/qwen', 
                         local_dir_use_symlinks=False)
        print('✅ Qwen model downloaded successfully')
    else:
        print('✅ Qwen model already exists locally')
except Exception as e:
    print(f'❌ Qwen model download failed: {e}')
    print('⚠️  Qwen will be downloaded at runtime')
" || echo "⚠️  Qwen будет загружен при первом запуске"

# Создание конфигурационного файла для образовательной среды
RUN echo "=== Создание образовательной конфигурации ===" && \
    mkdir -p /app/configs && \
    echo "EDUCATIONAL_MODE=true" > /app/configs/.env && \
    echo "PERFORMANCE_LEVEL=balanced" >> /app/configs/.env && \
    echo "ENABLE_EXPERIMENTAL_FEATURES=true" >> /app/configs/.env && \
    echo "MAX_FILE_SIZE_MB=1000" >> /app/configs/.env && \
    echo "✅ Educational configuration created"

# Копирование файлов приложения
COPY task_manager.py /app/
COPY main.py /app/
COPY config.py /app/
COPY exceptions.py /app/
COPY types.py /app/
COPY processing/ /app/processing/
COPY static/ /app/static/
COPY EDUCATIONAL_GUIDE.md /app/

# Создание скрипта запуска для образовательной среды
RUN echo '#!/bin/bash' > /app/start_educational.sh && \
    echo 'echo "🎓 Запуск системы для образовательной среды..."' >> /app/start_educational.sh && \
    echo 'echo "📚 Документация: http://localhost:8000/educational-setup"' >> /app/start_educational.sh && \
    echo 'echo "🔧 Рекомендации: http://localhost:8000/performance-recommendations"' >> /app/start_educational.sh && \
    echo 'echo "📊 Анализ системы: http://localhost:8000/system-capacity"' >> /app/start_educational.sh && \
    echo 'echo "⚡ Запуск uvicorn..."' >> /app/start_educational.sh && \
    echo 'exec uvicorn main:app --host 0.0.0.0 --port 8000' >> /app/start_educational.sh && \
    chmod +x /app/start_educational.sh

# Информация о системе
RUN echo "=== Информация о системе ===" && \
    python3 -c "
import torch
print(f'🔥 PyTorch version: {torch.__version__}')
print(f'🚀 CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎯 CUDA version: {torch.version.cuda}')
    print(f'📊 GPU: {torch.cuda.get_device_name(0)}')
    print(f'💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print('✅ System ready for educational use!')
"

EXPOSE 8000

# Команда запуска с образовательными настройками
CMD ["/app/start_educational.sh"]