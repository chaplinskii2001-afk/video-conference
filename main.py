import os
import uuid
import asyncio
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from task_manager import task_manager
import aiofiles
import torch
from processing.video_processor import VideoProcessor

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


app = FastAPI(title="Video Conference Processor", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("GET /progress") == -1


# Применяем фильтр к логгеру uvicorn.access
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# Монтирование статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    """Главная страница"""
    return FileResponse("static/index.html")


@app.get("/memory-status")
async def memory_status():
    """Проверка использования памяти"""
    try:
        if not processor:
            raise HTTPException(500, "Processor not initialized")

        gpu_info = processor.get_gpu_memory_info()
        system_info = processor.get_system_info()

        return {
            "gpu_memory": gpu_info,
            "system_info": system_info,
            "current_loaded_model": processor.current_loaded_model,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/model-status")
async def model_status():
    """Статус загруженных AI моделей"""
    try:
        if not processor:
            raise HTTPException(500, "Processor not initialized")

        # Получаем информацию о моделях через новый API
        models_info = {}
        if hasattr(processor, 'model_manager'):
            models_info = processor.model_manager.get_loaded_models_info()
        
        system_info = processor.get_system_info()

        return {
            "models": models_info,
            "system": system_info
        }
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
    """Endpoint для запуска обработки медиа"""
    if summary_type not in ["standard", "protocol"]:
        summary_type = "standard"
    try:
        task_id = str(uuid.uuid4())
        task_manager.create_task(task_id)

        # Определяем тип медиа
        media_type = "video"
        if file:
            file_ext = os.path.splitext(file.filename)[1].lower()
            audio_extensions = [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"]
            if file_ext in audio_extensions:
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
        }

    except Exception as e:
        logger.error(f"Ошибка запуска обработки: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
    try:
        gpu_available = torch.cuda.is_available()
        gpu_info = {}

        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_info = {
                "gpu_count": gpu_count,
                "current_device": torch.cuda.current_device(),
                "device_name": (
                    torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                ),
            }

        return {"gpu_available": gpu_available, "gpu_info": gpu_info}
    except Exception as e:
        return {"gpu_available": False, "error": str(e)}
