# Video Conference Processor

Веб-приложение для автоматической обработки видео и аудио конференций с использованием AI моделей. Приложение выполняет транскрипцию речи, диаризацию спикеров и создание краткого содержания.

## 🎯 Возможности

- **Транскрипция**: Преобразование речи в текст с использованием Whisper (русский язык)
- **Диаризация**: Автоматическое определение спикеров (кто и когда говорил)
- **Суммаризация**: Создание краткого содержания или протокола конференции
- **Поддержка форматов**: MP4, WebM, MOV, AVI, MKV, MP3, WAV, M4A, FLAC, AAC, OGG
- **Загрузка по URL**: Поддержка YouTube и других платформ через yt-dlp

## 🏗️ Архитектура

- **Backend**: FastAPI (Python 3.10)
- **Frontend**: HTML + JavaScript (Vanilla)
- **AI модели**:
  - Whisper (bond005/whisper-podlodka-turbo) - транскрипция
  - SpeechBrain (spkrec-ecapa-voxceleb) - диаризация
  - Qwen3-4B-Instruct - суммаризация
- **Инфраструктура**: Docker + Docker Compose
- **GPU**: Nvidia CUDA 12.2 (требуется GPU с минимум 6GB VRAM)

## 📋 Требования

### Системные требования

- Ubuntu 20.04+ (или другой Linux дистрибутив)
- Nvidia GPU с минимум 6GB VRAM
- Nvidia Driver 525+
- Docker 20.10+
- Docker Compose v2+
- Nvidia Container Toolkit

### Установка Nvidia Container Toolkit

```bash
# Добавить репозиторий
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Установить
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Перезапустить Docker
sudo systemctl restart docker
```

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/video-conference-processor.git
cd video-conference-processor
```

### 2. Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```bash
HF_TOKEN=your_huggingface_token_here
```

> **Примечание**: Получите токен на [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 3. Сборка Docker образа

```bash
docker-compose build
```

> ⚠️ **Важно**: Сборка может занять 20-40 минут, так как скачиваются AI модели (~15GB)

### 4. Запуск приложения

```bash
docker-compose up -d
```

### 5. Проверка работы

Откройте в браузере: [http://localhost:8000](http://localhost:8000)

Проверка здоровья сервера:
```bash
curl http://localhost:8000/health
```

## 📊 API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | Главная страница |
| `/process` | POST | Запуск обработки медиа |
| `/progress/{task_id}` | GET | Получение прогресса задачи |
| `/download/{task_id}/summary` | GET | Скачать краткое содержание |
| `/download/{task_id}/transcription` | GET | Скачать полную расшифровку |
| `/health` | GET | Проверка здоровья сервера |
| `/gpu-status` | GET | Статус GPU |
| `/memory-status` | GET | Использование памяти |

## 🔧 Конфигурация

### docker-compose.yml

Основные параметры:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0              # Использовать первую GPU
  - HF_TOKEN=${HF_TOKEN}                # Токен HuggingFace
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32  # Оптимизация памяти
```

### Порты

- `8000` - HTTP сервер (FastAPI)

### Volumes

- `./logs` - логи приложения
- `./uploads` - временные загруженные файлы
- `./results` - результаты обработки

## 📝 Использование

### Через веб-интерфейс

1. Откройте http://localhost:8000
2. Выберите файл или вставьте URL
3. Выберите тип суммаризации (обычная или протокол)
4. Нажмите "Обработать"
5. Дождитесь завершения и скачайте результаты

### Через API

```bash
# Загрузка файла
curl -X POST http://localhost:8000/process \
  -F "file=@video.mp4" \
  -F "summary_type=standard"

# Получение task_id
# {"task_id": "abc-123", "status": "processing"}

# Проверка прогресса
curl http://localhost:8000/progress/abc-123

# Скачивание результатов
curl http://localhost:8000/download/abc-123/summary -o summary.md
curl http://localhost:8000/download/abc-123/transcription -o transcription.md
```

## 🛠️ Разработка

### Структура проекта

```
video-conference-processor/
├── docker/
│   └── Dockerfile              # Конфигурация Docker образа
├── processing/
│   └── video_processor.py      # Логика обработки AI
├── static/
│   ├── index.html              # Frontend
│   ├── script.js               # JavaScript
│   └── style.css               # Стили
├── docker-compose.yml          # Docker Compose конфигурация
├── main.py                     # FastAPI сервер
├── task_manager.py             # Менеджер задач
└── requirements.txt            # Python зависимости
```

### Локальная разработка (без Docker)

```bash
# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Запустить сервер
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🐛 Troubleshooting

### GPU не обнаружена

```bash
# Проверить драйвер Nvidia
nvidia-smi

# Проверить Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

### Ошибка "CUDA out of memory"

- Убедитесь, что GPU имеет минимум 6GB VRAM
- Проверьте, что другие процессы не используют GPU
- Перезапустите контейнер: `docker-compose restart`

### Модели не загружаются

- Проверьте наличие HF_TOKEN в .env
- Пересоберите образ: `docker-compose build --no-cache`

## 📦 Зависимости

Основные библиотеки:

- `fastapi` - веб-фреймворк
- `torch` - PyTorch для AI моделей
- `transformers` - Whisper и Qwen модели
- `speechbrain` - диаризация
- `yt-dlp` - загрузка видео по URL
- `ffmpeg-python` - обработка аудио/видео

Полный список в [requirements.txt](requirements.txt)

## 🔒 Безопасность

- **Не коммитьте .env файл** с токенами
- Для production ограничьте CORS в `main.py`
- Добавьте rate limiting для API
- Используйте HTTPS (nginx reverse proxy)


