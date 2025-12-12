import os
import uuid
import asyncio
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from task_manager import task_manager
import aiofiles
import torch
from processing.video_processor import VideoProcessor
from config import get_settings, get_security_config, get_model_config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/app/logs/app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _get_tomsk_time():
    """Получение текущего времени в Томске (UTC+7)"""
    tomsk_tz = timezone(timedelta(hours=7))
    return datetime.now(tomsk_tz)


# Глобальные переменные
processor = None
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(
        "Запуск приложения... ({_get_tomsk_time().strftime('%Y-%m-%d %H:%M:%S')})"
    )
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("/app/logs", exist_ok=True)

    global processor
    processor = VideoProcessor()

    # Проверяем доступность моделей
    logger.info("Проверка доступности моделей...")

    model_dirs = {
        "whisper": "/app/models/whisper",
        "pyannote": "/app/models/pyannote",
        "qwen": "/app/models/qwen",
    }

    for model_name, model_path in model_dirs.items():
        if os.path.exists(model_path):
            logger.info(f"✅ Модель {model_name} найдена в {model_path}")
        else:
            logger.warning(f"⚠️ Модель {model_name} не найдена в {model_path}")

    logger.info("Все модели будут загружаться динамически по мере необходимости")
    logger.info("Qwen модель будет загружаться с 4-битным квантованием")

    yield  # Здесь приложение работает

    # Shutdown
    logger.info(
        "Остановка приложения... ({_get_tomsk_time().strftime('%Y-%m-%d %H:%M:%S')})"
    )
    if processor:
        await processor.force_gpu_cleanup()
        await processor.unload_current_model()
    logger.info("Приложение остановлено")


# Получаем настройки
settings = get_settings()
security_config = get_security_config()
model_config = get_model_config()

app = FastAPI(title="Video Conference Processor", lifespan=lifespan)

# CORS - адаптировано для учебной/исследовательской среды
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("GET /progress") == -1


# Применяем фильтр к логгеру uvicorn.access
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# Монтирование статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/memory-status")
async def memory_status():
    """Проверка использования памяти"""
    try:
        if not processor:
            raise HTTPException(500, "Processor not initialized")

        gpu_info = processor.get_gpu_memory_info()
        system_info = processor.get_system_memory_info()

        return {
            "gpu_memory": gpu_info,
            "system_memory": system_info,
            "current_loaded_model": processor.current_loaded_model,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/model-status")
async def model_status():
    try:
        if not processor:
            raise HTTPException(500, "Processor not initialized")

        models_info = {
            "whisper_loaded": processor.whisper_model is not None,
            "whisper_model_size": getattr(processor, "whisper_model_size", "medium"),
            "diarization_loaded": processor.diarization_pipeline is not None,
            "qwen_loaded": processor.qwen_model is not None,
            "current_loaded_model": processor.current_loaded_model,
            "qwen_quantization": "4-bit",
        }

        if torch.cuda.is_available():
            gpu_info = {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
            }
        else:
            gpu_info = {"gpu_available": False}

        return {"models": models_info, "gpu": gpu_info}
    except Exception as e:
        return {"error": str(e)}


@app.get("/check-ffmpeg")
async def check_ffmpeg():
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ffmpeg_info = result.stdout.split("\n")[0] if result.stdout else "No output"

        return {
            "ffmpeg_available": result.returncode == 0,
            "ffmpeg_version": ffmpeg_info,
        }
    except Exception as e:
        return {"ffmpeg_available": False, "error": str(e)}


@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    url: str = Form(None),
    summary_type: str = Form("standard"),
):
    """
    Endpoint для запуска обработки медиа с валидацией файлов
    """
    # Валидация параметров
    if summary_type not in ["standard", "protocol"]:
        summary_type = "standard"
    
    if not file and not url:
        raise HTTPException(
            status_code=400, 
            detail="Необходимо предоставить файл или URL"
        )
    
    if file and url:
        raise HTTPException(
            status_code=400,
            detail="Нельзя одновременно загружать файл и указывать URL"
        )
    
    # Валидация файла
    if file:
        # Проверяем расширение
        if not security_config.validate_file_extension(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
            )
        
        # Проверяем размер (если возможно)
        if hasattr(file, 'size') and file.size:
            if not security_config.validate_file_size(file.size):
                raise HTTPException(
                    status_code=400,
                    detail=f"Размер файла превышает максимальный ({settings.MAX_FILE_SIZE_MB}MB)"
                )
    
    try:
        task_id = str(uuid.uuid4())
        task_manager.create_task(task_id)

        # Определяем тип медиа
        media_type = "video"
        if file:
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext in settings.ALLOWED_AUDIO_EXTENSIONS:
                media_type = "audio"

        # Запускаем обработку в фоне
        background_tasks.add_task(
            process_media_background,
            task_id,
            file if file else None,
            url,
            media_type,
            summary_type,
        )

        logger.info(
            f"Задача {task_id} запущена в фоне (тип суммаризации: {summary_type})"
        )

        return {
            "task_id": task_id,
            "status": "processing",
            "summary_type": summary_type,
            "media_type": media_type,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка запуска обработки: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


async def process_media_background(
    task_id: str,
    file: UploadFile = None,
    url: str = None,
    media_type: str = "video",
    summary_type: str = "standard",
):
    """Фоновая обработка медиа"""
    processor = VideoProcessor(task_id=task_id)
    file_path = None

    try:
        # Обновляем прогресс
        task_manager.update_progress(
            task_id, 5, "download", f"Начало обработки {media_type}"
        )

        # Сохраняем файл или скачиваем по URL
        if file:
            file_ext = os.path.splitext(file.filename)[1].lower()
            file_path = f"uploads/{task_id}{file_ext}"

            task_manager.update_progress(
                task_id, 10, "download", f"Сохранение файла: {file.filename}"
            )

            async with aiofiles.open(file_path, "wb") as f:
                content = await file.read()
                await f.write(content)

            logger.info(f"Файл сохранен: {file_path}")

        elif url:
            task_manager.update_progress(
                task_id, 10, "download", f"Скачивание по URL: {url}"
            )
            file_path = await processor.download_from_url(url, task_id)
        else:
            raise ValueError("Не предоставлен файл или URL")

        # Обрабатываем медиа
        task_manager.update_progress(
            task_id,
            15,
            "preprocessing",
            f"Подготовка к обработке {media_type} (суммаризация: {summary_type})",
        )

        result = await processor.process_media(
            file_path, task_id, media_type, summary_type
        )

        # Сохраняем результат
        task_manager.complete_task(task_id, result)
        logger.info(f"Задача {task_id} завершена успешно")

    except Exception as e:
        logger.error(f"Ошибка обработки задачи {task_id}: {str(e)}", exc_info=True)
        task_manager.fail_task(task_id, str(e))

    finally:
        # Очистка временных файлов
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Временный файл удален: {file_path}")
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл {file_path}: {e}")


@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """Получение прогресса по задаче"""
    task_info = task_manager.get_task_info(task_id)

    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": task_info["status"],
        "percent": task_info["percent"],
        "current_stage": task_info["current_stage"],
        "logs": task_info["logs"][-10:],  # Последние 10 логов
        "result": task_info.get("result"),
        "error": task_info.get("error"),
    }


@app.get("/download/{task_id}/{file_type}")
async def download_file(task_id: str, file_type: str):
    """Скачивание результатов"""
    logger.info(f"Запрос на скачивание: task_id={task_id}, file_type={file_type}")

    if file_type == "summary":
        filename = f"{task_id}_summary.md"
        file_path = f"{RESULTS_DIR}/{filename}"
    elif file_type == "transcription":
        filename = f"{task_id}_transcription.md"
        file_path = f"{RESULTS_DIR}/{filename}"
    else:
        logger.error(f"Неизвестный тип файла: {file_type}")
        raise HTTPException(404, "Тип файла не поддерживается")

    # Проверяем существование файла
    if not os.path.exists(file_path):
        logger.error(f"Файл не найден: {file_path}")
        # Проверим, есть ли файлы в директории results
        try:
            files = os.listdir(RESULTS_DIR)
            logger.info(f"Файлы в директории results: {files}")
        except Exception as e:
            logger.error(f"Ошибка при чтении директории results: {e}")
        raise HTTPException(404, "Файл не найден")

    logger.info(f"Файл найден, отправка: {file_path}")
    return FileResponse(path=file_path, media_type="text/markdown", filename=filename)


# Периодическая очистка старых задач
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_worker())


async def cleanup_worker():
    while True:
        await asyncio.sleep(3600)  # Каждый час
        task_manager.cleanup_old_tasks()


@app.get("/health")
async def health_check():
    logger.info("Проверка здоровья приложения")
    gpu_status = processor.check_gpu() if processor else False

    dirs_status = {
        "uploads": os.path.exists(UPLOAD_DIR),
        "results": os.path.exists(RESULTS_DIR),
        "models_whisper": os.path.exists("/app/models/whisper"),
        "models_qwen": os.path.exists("/app/models/qwen"),
    }

    return {
        "status": "healthy",
        "gpu_available": gpu_status,
        "directories": dirs_status,
        "server_time_tomsk": _get_tomsk_time().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/gpu-status")
async def gpu_status():
    """Детальная информация о GPU для исследовательских целей"""
    try:
        from config import detect_gpu_capabilities, get_performance_config
        
        gpu_capabilities = detect_gpu_capabilities()
        gpu_available = gpu_capabilities["gpu_available"]
        
        if gpu_available:
            perf_config = get_performance_config(gpu_capabilities["memory_gb"])
            gpu_info = {
                "capabilities": gpu_capabilities,
                "recommended_performance": perf_config,
                "suitable_for_research": gpu_capabilities.get("suitable_for_research", False),
            }
        else:
            gpu_info = {
                "gpu_available": False,
                "recommendation": "Рекомендуется использовать GPU для исследований",
            }

        return {"gpu_available": gpu_available, "gpu_info": gpu_info}
    except Exception as e:
        return {"gpu_available": False, "error": str(e)}


@app.get("/educational-setup")
async def educational_setup():
    """Информация о настройках для учебной/исследовательской среды"""
    try:
        from config import get_educational_setup
        
        setup_info = get_educational_setup()
        return {
            "educational_mode": True,
            "setup_info": setup_info,
            "recommended_usage": _get_usage_recommendations(setup_info),
        }
    except Exception as e:
        return {"educational_mode": True, "error": str(e)}


@app.get("/performance-recommendations")
async def performance_recommendations():
    """Рекомендации по оптимизации производительности"""
    try:
        from config import detect_gpu_capabilities, get_performance_config
        
        gpu_info = detect_gpu_capabilities()
        if gpu_info["gpu_available"]:
            perf_config = get_performance_config(gpu_info["memory_gb"])
            return {
                "current_hardware": gpu_info,
                "recommended_settings": perf_config,
                "optimization_tips": _get_optimization_tips(gpu_info["memory_gb"]),
            }
        else:
            return {
                "current_hardware": gpu_info,
                "recommended_settings": "cpu_only",
                "optimization_tips": [
                    "Рекомендуется использовать GPU для ускорения",
                    "Уменьшите размер обрабатываемых файлов",
                    "Используйте квантование моделей для экономии памяти"
                ],
            }
    except Exception as e:
        return {"error": str(e)}


@app.get("/system-capacity")
async def system_capacity():
    """Анализ системной производительности для исследований"""
    try:
        from config import detect_gpu_capabilities
        import psutil
        
        gpu_info = detect_gpu_capabilities()
        cpu_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        }
        
        # Анализ пригодности для исследований
        research_capacity = _analyze_research_capacity(gpu_info, cpu_info)
        
        return {
            "gpu_info": gpu_info,
            "cpu_info": cpu_info,
            "research_capacity": research_capacity,
            "recommendations": research_capacity["recommendations"],
        }
    except Exception as e:
        return {"error": str(e)}


def _get_usage_recommendations(setup_info: dict) -> dict:
    """Генерация рекомендаций по использованию"""
    hardware_info = setup_info["hardware_info"]
    
    recommendations = {
        "file_size_recommendations": [],
        "processing_tips": [],
        "research_suggestions": [],
    }
    
    if hardware_info["gpu_available"]:
        memory_gb = hardware_info["memory_gb"]
        
        if memory_gb >= 16:
            recommendations["file_size_recommendations"].extend([
                "Можно обрабатывать файлы до 1GB",
                "Поддерживается пакетная обработка",
                "Рекомендуется высокое качество моделей"
            ])
            recommendations["processing_tips"].extend([
                "Используйте параллельную обработку",
                "Включите экспериментальные функции",
                "Сохраняйте промежуточные результаты"
            ])
        elif memory_gb >= 8:
            recommendations["file_size_recommendations"].extend([
                "Рекомендуется файлы до 500MB",
                "Лучше обрабатывать последовательно"
            ])
            recommendations["processing_tips"].extend([
                "Используйте квантование 8-bit для Whisper",
                "Контролируйте использование памяти"
            ])
        else:
            recommendations["file_size_recommendations"].extend([
                "Ограничьтесь файлами до 100MB",
                "Обрабатывайте короткие записи"
            ])
    else:
        recommendations["file_size_recommendations"].extend([
            "Рекомендуется использовать только для простых задач",
            "Файлы должны быть короткими (до 30 минут)"
        ])
    
    return recommendations


def _get_optimization_tips(memory_gb: float) -> list:
    """Советы по оптимизации производительности"""
    tips = []
    
    if memory_gb >= 24:
        tips.extend([
            "Используйте модели без квантования для лучшего качества",
            "Включите пакетную обработку",
            "Экспериментируйте с разными параметрами модели"
        ])
    elif memory_gb >= 12:
        tips.extend([
            "Используйте 8-bit квантование для баланса качества/производительности",
            "Оптимально для большинства исследовательских задач"
        ])
    elif memory_gb >= 8:
        tips.extend([
            "Обязательно используйте квантование моделей",
            "Обрабатывайте файлы последовательно",
            "Мониторьте использование памяти"
        ])
    else:
        tips.extend([
            "Используйте только для демонстрации и обучения",
            "Ограничьтесь короткими файлами (до 30 минут)",
            "Готовьтесь к длительному времени обработки"
        ])
    
    return tips


def _analyze_research_capacity(gpu_info: dict, cpu_info: dict) -> dict:
    """Анализ пригодности для исследований"""
    capacity = {
        "overall_rating": "poor",
        "suitable_for_research": False,
        "recommendations": [],
        "limitations": [],
    }
    
    gpu_memory = gpu_info.get("memory_gb", 0)
    cpu_cores = cpu_info.get("cpu_count", 1)
    total_memory = cpu_info.get("memory_total_gb", 0)
    
    # Оценка общей производительности
    if gpu_memory >= 24 and cpu_cores >= 8 and total_memory >= 32:
        capacity["overall_rating"] = "excellent"
        capacity["suitable_for_research"] = True
        capacity["recommendations"].extend([
            "Отлично подходит для серьезных исследований",
            "Можно использовать для обучения моделей",
            "Поддерживает пакетную обработку"
        ])
    elif gpu_memory >= 12 and cpu_cores >= 6 and total_memory >= 16:
        capacity["overall_rating"] = "good"
        capacity["suitable_for_research"] = True
        capacity["recommendations"].extend([
            "Хорошо подходит для большинства исследований",
            "Рекомендуется использовать квантование моделей"
        ])
    elif gpu_memory >= 8 and cpu_cores >= 4 and total_memory >= 8:
        capacity["overall_rating"] = "acceptable"
        capacity["suitable_for_research"] = True
        capacity["recommendations"].extend([
            "Минимальные требования для исследований",
            "Используйте последовательную обработку",
            "Ограничьте размер файлов"
        ])
        capacity["limitations"].extend([
            "Ограниченная производительность",
            "Не подходит для обучения моделей"
        ])
    else:
        capacity["overall_rating"] = "limited"
        capacity["recommendations"].extend([
            "Подходит только для демонстрации и обучения",
            "Для серьезных исследований требуется более мощное железо"
        ])
        capacity["limitations"].extend([
            "Низкая производительность",
            "Длительное время обработки",
            "Ограниченные возможности для исследований"
        ])
    
    return capacity
