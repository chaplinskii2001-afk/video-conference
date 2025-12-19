FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ARG HF_TOKEN

# Устанавливаем переменные окружения для CUDA
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

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
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Установка полного набора библиотек cuDNN 9 для CUDA 12
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/models/whisper /app/models/pyannote /app/models/qwen /app/logs /app/uploads /app/results

RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install "pyannote.audio @ git+https://github.com/pyannote/pyannote-audio.git" \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --no-cache-dir

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install pynvml psutil huggingface_hub

# 1. Whisper (faster-whisper / CTranslate2) - ПРИНУДИТЕЛЬНО С ПРАВИЛЬНОЙ МОДЕЛЬЮ
RUN echo "🔧 ПРИНУДИТЕЛЬНАЯ ЗАГРУЗКА МОДЕЛИ Systran/faster-whisper-large-v3" && \
    rm -rf /app/models/whisper/* && \
    python3 -c "from faster_whisper import WhisperModel; \
    print('Loading Systran/faster-whisper-large-v3...'); \
    model = WhisperModel('Systran/faster-whisper-large-v3', device='cpu', compute_type='int8', download_root='/app/models/whisper'); \
    print('✅ faster-whisper model downloaded successfully'); \
    import os; print('Contents:', os.listdir('/app/models/whisper'))"

# Silero VAD (cached via torch.hub)
RUN python3 -c "import torch; \
    torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False); \
    print('Silero VAD downloaded')"

# 2. PyAnnote
RUN python3 -c "import torch; import torchaudio; \
    from pyannote.audio import Pipeline; \
    hf_token = '$HF_TOKEN'; \
    Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token=hf_token, cache_dir='/app/models/pyannote'); \
    print('PyAnnote model downloaded')"

# 3. Qwen
RUN python3 -c "from huggingface_hub import snapshot_download; \
    import os; \
    snapshot_download(repo_id='Qwen/Qwen3-4B-Instruct-2507', local_dir='/app/models/qwen', local_dir_use_symlinks=False); \
    print('Qwen model downloaded successfully')"

# Проверка
RUN echo "=== Проверка моделей ===" && \
    echo "Whisper:" && find /app/models/whisper/ -name "*.json" | head -1 && \
    echo "PyAnnote:" && find /app/models/pyannote/ -maxdepth 2 | head -5 && \
    echo "Qwen:" && [ -f /app/models/qwen/config.json ] && echo "✅ Qwen config exists" || echo "❌ Qwen config missing"

COPY task_manager.py /app/
COPY main.py /app/
COPY processing/ /app/processing/
COPY static/ /app/static/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
