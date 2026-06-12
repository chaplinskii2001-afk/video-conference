FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# 1) Единые директории кэша/моделей внутри образа (будут переопределены через volume в docker-compose)
ENV HF_HOME=/app/models/hf_home
ENV TRANSFORMERS_CACHE=/app/models/hf_home/transformers
ENV HF_HUB_CACHE=/app/models/hf_home/hub
ENV XDG_CACHE_HOME=/app/models/xdg_cache
ENV TORCH_HOME=/app/models/torch_home

RUN rm -f /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --no-install-recommends \
    ca-certificates ubuntu-keyring gnupg \
    python3.10 python3-pip libpython3.10 \
    ffmpeg git git-lfs wget curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

# Создаем базовые директории (модели и кэш будут примонтированы через volume с хоста)
RUN mkdir -p /app/models/whisper /app/models/pyannote /app/models/qwen \
    /app/logs /app/uploads /app/results \
    $HF_HOME $TRANSFORMERS_CACHE $HF_HUB_CACHE $XDG_CACHE_HOME $TORCH_HOME

# Установка Python-зависимостей
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install "pyannote.audio @ git+https://github.com/pyannote/pyannote-audio.git" \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --no-cache-dir

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# КРИТИЧНО: Принудительное обновление transformers для поддержки архитектуры Qwen3 
# (Версии < 4.51.0 вызывают KeyError: 'qwen3' при загрузке модели)
RUN pip3 install --no-cache-dir --upgrade "transformers>=4.51.0"

RUN pip3 install --no-cache-dir pynvml psutil huggingface_hub

# Копирование исходного кода
COPY task_manager.py /app/
COPY main.py /app/
COPY processing/ /app/processing/
COPY static/ /app/static/
COPY config.py /app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
