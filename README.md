# 🎓 Video Conference Processor - Образовательная версия

Веб-приложение для автоматической обработки видео и аудио конференций с использованием AI моделей. **Специально адаптировано для учебной и исследовательской среды** с дополнительными возможностями для изучения современных технологий искусственного интеллекта.

## 🎯 Возможности

### Основной функционал
- **Транскрипция**: Преобразование речи в текст с использованием Whisper (русский язык)
- **Диаризация**: Автоматическое определение спикеров (кто и когда говорил)
- **Суммаризация**: Создание краткого содержания или протокола конференции
- **Поддержка форматов**: MP4, WebM, MOV, AVI, MKV, MP3, WAV, M4A, FLAC, AAC, OGG
- **Загрузка по URL**: Поддержка YouTube и других платформ через yt-dlp

### 🎓 Образовательные возможности
- **Автоматическое определение возможностей системы**: анализ GPU и рекомендации настроек
- **Адаптивная производительность**: автоматическая оптимизация под доступное железо
- **Образовательные API endpoints**: для изучения и анализа
- **Расширенные сообщения об ошибках**: с советами для обучения
- **Мониторинг производительности**: детальная аналитика для исследований
- **Поддержка больших файлов**: до 1GB для исследовательских задач

## 🏗️ Архитектура

### Основные компоненты
- **Backend**: FastAPI (Python 3.10) с образовательными endpoints
- **Frontend**: HTML + JavaScript (Vanilla)
- **AI модели**:
  - Whisper (bond005/whisper-podlodka-turbo) - транскрипция
  - PyAnnote (speaker-diarization-3.1) - диаризация  
  - Qwen3-4B-Instruct - суммаризация
- **Инфраструктура**: Docker + Docker Compose (образовательная версия)
- **GPU**: Nvidia CUDA 12.4 (требуется GPU с минимум 6GB VRAM)

### 🎓 Образовательные компоненты
- **Автоматическая конфигурация**: система определяет возможности GPU и настраивается автоматически
- **Образовательные API endpoints**: для изучения и анализа системы
- **Расширенный мониторинг**: детальная аналитика производительности
- **Гибкая архитектура**: готова к масштабированию на более мощное железо

## 📋 Требования

### 🎓 Системные требования для образовательной среды

**Минимальные требования:**
- Ubuntu 20.04+ (или другой Linux дистрибутив)
- Nvidia GPU с минимум 6GB VRAM
- Nvidia Driver 525+
- Docker 20.10+
- Docker Compose v2+
- Nvidia Container Toolkit

**Рекомендуемые конфигурации для исследований:**
- **Мощная система**: 24+ GB GPU, 32+ GB RAM, 8+ CPU cores
- **Средняя система**: 12-16 GB GPU, 16+ GB RAM, 6+ CPU cores  
- **Бюджетная система**: 8-12 GB GPU, 8+ GB RAM, 4+ CPU cores

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

## 🚀 Быстрый старт для образовательной среды

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/video-conference-processor.git
cd video-conference-processor
```

### 2. Настройка для образовательной среды

Скопируйте файл конфигурации и настройте под вашу систему:

```bash
# Копируем пример конфигурации
cp .env.example .env

# Редактируем настройки под вашу систему
nano .env
```

> **Для образовательной среды рекомендуется:**
> - Установить `EDUCATIONAL_MODE=true`
> - Настроить `PERFORMANCE_LEVEL` под ваш GPU
> - Указать ваш `HF_TOKEN`

### 3. Запуск образовательной системы

```bash
# Полная сборка и запуск (первый раз)
docker-compose up --build

# Или только запуск (если образ уже собран)
docker-compose up -d
```

> ⚠️ **Важно**: Первая сборка может занять 30-60 минут, так как скачиваются AI модели (~15GB)

### 4. Проверка образовательной системы

**Основной интерфейс:** [http://localhost:8000](http://localhost:8000)

**Образовательные возможности:**
- **Анализ системы**: http://localhost:8000/educational-setup
- **Рекомендации производительности**: http://localhost:8000/performance-recommendations  
- **Возможности системы**: http://localhost:8000/system-capacity
- **Мониторинг GPU**: http://localhost:8000/gpu-status

### 5. Мониторинг для исследований

```bash
# Логи приложения
docker-compose logs -f video-processor

# Мониторинг GPU (если включен)
docker-compose logs -f gpu-monitor

# Проверка здоровья
curl http://localhost:8000/health
```

## 📊 API Endpoints

### Основные API
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | Главная страница |
| `/process` | POST | Запуск обработки медиа |
| `/progress/{task_id}` | GET | Получение прогресса задачи |
| `/download/{task_id}/summary` | GET | Скачать краткое содержание |
| `/download/{task_id}/transcription` | GET | Скачать полную расшифровку |
| `/health` | GET | Проверка здоровья сервера |

### 🎓 Образовательные API endpoints
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/gpu-status` | GET | Детальный статус GPU с рекомендациями |
| `/memory-status` | GET | Использование памяти GPU и системы |
| `/model-status` | GET | Статус загрузки AI моделей |
| `/check-ffmpeg` | GET | Проверка установки ffmpeg |
| `/educational-setup` | GET | Настройки для образовательной среды |
| `/performance-recommendations` | GET | Рекомендации по оптимизации |
| `/system-capacity` | GET | Анализ пригодности для исследований |

### Мониторинг и аналитика
- **GPU мониторинг**: Детальная информация о видеокарте
- **Системная аналитика**: CPU, RAM, производительность
- **Образовательные рекомендации**: Персонализированные советы

## 🎓 Образовательные возможности

### Автоматическая оптимизация под оборудование

Система автоматически определяет возможности вашего GPU и настраивает оптимальные параметры:

```python
# Пример автоматического определения
{
    "gpu_memory_gb": 12.0,
    "suitable_for_research": true,
    "recommended_settings": {
        "whisper_quantization": "8bit",
        "qwen_quantization": "4bit", 
        "max_concurrent_tasks": 2,
        "chunk_size_multiplier": 1.0
    }
}
```

### Исследовательские сценарии

**1. Анализ качества транскрипции**
- Тестирование с аудио разного качества
- Сравнение точности на коротких и длинных записях
- Анализ влияния шума на качество

**2. Оптимизация производительности**
- Тестирование различных настроек квантования
- Анализ использования памяти GPU
- Оптимизация скорости обработки

**3. Изучение архитектуры AI**
- Анализ моделей Whisper, PyAnnote, Qwen
- Изучение квантования нейронных сетей
- Практика работы с GPU и CUDA

### Образовательные API для исследований

```bash
# Анализ пригодности системы для исследований
curl http://localhost:8000/system-capacity

# Получение рекомендаций по настройке
curl http://localhost:8000/performance-recommendations

# Детальная информация о GPU
curl http://localhost:8000/gpu-status
```

## 🔧 Конфигурация для образовательной среды

### Основные параметры в .env

```bash
# Образовательные настройки
EDUCATIONAL_MODE=true
PERFORMANCE_LEVEL=balanced  # research/balanced/performance
ENABLE_EXPERIMENTAL_FEATURES=true

# Для исследований
MAX_FILE_SIZE_MB=1000  # Увеличено для образовательной среды
MAX_PROCESSING_TIME_HOURS=4
SAVE_INTERMEDIATE_RESULTS=true
DETAILED_LOGGING=true
```

### Рекомендуемые настройки для разных GPU

**Для мощных GPU (24+ GB):**
```bash
PERFORMANCE_LEVEL=research
MAX_FILE_SIZE_MB=2000
```

**Для средних GPU (12-16 GB):**
```bash
PERFORMANCE_LEVEL=balanced  
MAX_FILE_SIZE_MB=1000
```

**Для бюджетных GPU (8-12 GB):**
```bash
PERFORMANCE_LEVEL=performance
MAX_FILE_SIZE_MB=500
```

### Порты

- `8000` - Основной HTTP сервер (FastAPI)
- `8080` - Jupyter notebook (опционально)

### Volumes для исследований

- `./logs` - логи приложения
- `./uploads` - временные загруженные файлы
- `./results` - результаты обработки
- `./data` - экспериментальные данные
- `./experiments` - результаты исследований
- `./notebooks` - Jupyter notebooks для анализа

## 📝 Использование

### Через веб-интерфейс

1. Откройте http://localhost:8000
2. **Образовательная среда**: Система автоматически определит возможности вашего оборудования
3. Выберите файл или вставьте URL (до 1GB для исследований)
4. Выберите тип суммаризации (обычная или протокол)
5. **Для исследований**: Можете использовать экспериментальные функции
6. Нажмите "Обработать" и отслеживайте детальный прогресс
7. Скачайте результаты и промежуточные файлы (если включено)

### 🎓 Через API для исследований

```bash
# Базовое использование
curl -X POST http://localhost:8000/process \
  -F "file=@conference_video.mp4" \
  -F "summary_type=standard"

# Получение детального анализа системы
curl http://localhost:8000/educational-setup

# Анализ производительности для исследований  
curl http://localhost:8000/system-capacity

# Рекомендации по оптимизации
curl http://localhost:8000/performance-recommendations

# Мониторинг обработки
curl http://localhost:8000/progress/{task_id}

# Скачивание результатов
curl http://localhost:8000/download/{task_id}/summary -o research_summary.md
curl http://localhost:8000/download/{task_id}/transcription -o full_transcription.md
```

### Исследовательские сценарии

**1. Анализ эффективности различных настроек:**
```bash
# Тестирование с разными типами квантования
curl http://localhost:8000/performance-recommendations

# Сравнение времени обработки разных файлов
# Анализ качества транскрипции на разном оборудовании
```

**2. Изучение архитектуры AI моделей:**
```bash
# Получение информации о загруженных моделях
curl http://localhost:8000/model-status

# Мониторинг использования GPU памяти
curl http://localhost:8000/memory-status
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

## 🎓 Специально для образовательной среды

### Целевое использование

Эта система разработана специально для:
- **Дипломных работ** студентов IT-специальностей
- **Исследований** в области обработки речи и AI
- **Обучения** современным технологиям машинного обучения
- **Демонстрации** возможностей современных языковых моделей

### Особенности для образования

- **Автоматическая адаптация** под доступное оборудование
- **Образовательные API endpoints** для изучения системы
- **Подробные рекомендации** по оптимизации под конкретный GPU
- **Расширенная документация** с объяснениями архитектуры
- **Готовность к масштабированию** на более мощное железо

### Получение помощи

- **Документация**: Читайте EDUCATIONAL_GUIDE.md
- **API документация**: Используйте образовательные endpoints
- **Логи**: Просматривайте детальные логи в `./logs/`
- **Мониторинг**: Используйте GPU мониторинг для исследований

## 🐛 Troubleshooting для образовательной среды

### GPU не обнаружена

```bash
# Проверить драйвер Nvidia
nvidia-smi

# Проверить Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi

# Для образовательной среды - анализ системы
curl http://localhost:8000/system-capacity
```

### Ошибка "CUDA out of memory"

- **Для образования**: Используйте систему рекомендаций
```bash
curl http://localhost:8000/performance-recommendations
```
- Уменьшите размер файлов или используйте более агрессивное квантование
- Проверьте другие процессы: `nvidia-smi`
- Перезапустите: `docker-compose restart`

### Модели не загружаются

- Проверьте HF_TOKEN в .env файле
- Для образовательной среды - проверьте автоматическую загрузку:
```bash
curl http://localhost:8000/model-status
```
- Пересоберите образ: `docker-compose build --no-cache`

### Образовательная отладка

```bash
# Анализ всей системы для исследований
curl http://localhost:8000/educational-setup

# Полные логи для анализа проблем
docker-compose logs -f video-processor

# Мониторинг GPU для исследования производительности
docker-compose logs -f gpu-monitor
```

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


