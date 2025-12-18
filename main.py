import os
import uuid
import asyncio
import logging
import subprocess
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
    Header,
    Query,
)
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

# По умолчанию сериализуем обработку (GPU-ограничение). В будущем можно увеличить.
processing_lock = asyncio.Lock()


def _resolve_client_id(
    *, x_client_id: Optional[str] = None, client_id: Optional[str] = None
) -> str:
    resolved = x_client_id or client_id
    if not resolved:
        raise HTTPException(status_code=400, detail="Missing client_id")

    if any(part in resolved for part in ("..", "/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid client_id")

    return resolved


def _ensure_user_dirs(user_id: str):
    os.makedirs(os.path.join(UPLOAD_DIR, user_id), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, user_id), exist_ok=True)


def _detect_media_type(filename: str) -> str:
    file_ext = os.path.splitext(filename)[1].lower()
    audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}
    return "audio" if file_ext in audio_extensions else "video"


async def _save_upload_file(upload_file: UploadFile, destination_path: str):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    async with aiofiles.open(destination_path, "wb") as out:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            await out.write(chunk)

    try:
        await upload_file.close()
    except Exception:
        pass



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(
        "Запуск приложения... ({_get_tomsk_time().strftime('%Y-%m-%d %H:%M:%S')})"
    )
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("/app/logs", exist_ok=True)
    os.makedirs("/app/models/whisper", exist_ok=True)
    os.makedirs("/app/models/pyannote", exist_ok=True)
    os.makedirs("/app/models/qwen", exist_ok=True)

    global processor
    processor = VideoProcessor()

    # Проверяем доступность моделей и инициализируем их при необходимости
    logger.info("Проверка доступности моделей...")

    model_dirs = {
        "whisper": "/app/models/whisper",
        "pyannote": "/app/models/pyannote",
        "qwen": "/app/models/qwen",
    }

    for model_name, model_path in model_dirs.items():
        if os.path.exists(model_path) and os.listdir(model_path):
            logger.info(f"✅ Модель {model_name} найдена в {model_path}")
        else:
            logger.warning(f"⚠️ Модель {model_name} не найдена в {model_path}")
            logger.info(f"   Модель будет загружена при первом использовании...")

    logger.info("Инициализация менеджера моделей...")
    try:
        await processor.model_manager.ensure_models_ready()
    except Exception as e:
        logger.warning(f"Некритичная ошибка при инициализации моделей: {e}")

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


@app.post("/batch/process")
async def process_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(None),
    url: str = Form(None),
    summary_type: str = Form("standard"),
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
):
    """Запуск пакетной обработки (в указанном порядке)."""
    if summary_type not in ["standard", "protocol"]:
        summary_type = "standard"

    client_id = x_client_id or str(uuid.uuid4())
    _ensure_user_dirs(client_id)

    try:
        if files and len(files) > 0:
            items_meta = [
                {"file_name": f.filename, "source_type": "file"} for f in files
            ]
            batch_id, batch_items = task_manager.create_batch(
                user_id=client_id, items=items_meta, summary_type=summary_type
            )

            for upload_file, item in zip(files, batch_items):
                task_id = item["task_id"]
                file_ext = os.path.splitext(upload_file.filename)[1].lower()
                file_path = os.path.join(
                    UPLOAD_DIR, client_id, f"{task_id}{file_ext}"
                )

                await _save_upload_file(upload_file, file_path)

                background_tasks.add_task(
                    process_media_background,
                    task_id=task_id,
                    user_id=client_id,
                    file_path=file_path,
                    url=None,
                    media_type=_detect_media_type(upload_file.filename),
                    summary_type=summary_type,
                )

            logger.info(
                f"Батч {batch_id} поставлен в очередь (user_id={client_id}, файлов={len(batch_items)})"
            )

            return {
                "client_id": client_id,
                "batch_id": batch_id,
                "status": "queued",
                "summary_type": summary_type,
                "items": batch_items,
            }

        if url:
            items_meta = [{"file_name": url, "source_type": "url"}]
            batch_id, batch_items = task_manager.create_batch(
                user_id=client_id, items=items_meta, summary_type=summary_type
            )
            task_id = batch_items[0]["task_id"]

            background_tasks.add_task(
                process_media_background,
                task_id=task_id,
                user_id=client_id,
                file_path=None,
                url=url,
                media_type="video",
                summary_type=summary_type,
            )

            logger.info(
                f"Батч {batch_id} поставлен в очередь (user_id={client_id}, url={url})"
            )

            return {
                "client_id": client_id,
                "batch_id": batch_id,
                "status": "queued",
                "summary_type": summary_type,
                "items": batch_items,
            }

        raise HTTPException(status_code=400, detail="Не предоставлены файлы или URL")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка запуска обработки: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    url: str = Form(None),
    summary_type: str = Form("standard"),
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
):
    """Совместимый endpoint для запуска обработки одного файла/URL."""
    if summary_type not in ["standard", "protocol"]:
        summary_type = "standard"

    if file:
        result = await process_batch(
            background_tasks,
            files=[file],
            url=None,
            summary_type=summary_type,
            x_client_id=x_client_id,
        )
    else:
        result = await process_batch(
            background_tasks,
            files=None,
            url=url,
            summary_type=summary_type,
            x_client_id=x_client_id,
        )

    task_id = result["items"][0]["task_id"]
    return {
        "client_id": result.get("client_id"),
        "batch_id": result.get("batch_id"),
        "task_id": task_id,
        "status": "processing",
        "summary_type": summary_type,
    }


async def process_media_background(
    *,
    task_id: str,
    user_id: str,
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    media_type: str = "video",
    summary_type: str = "standard",
):
    """Фоновая обработка медиа."""
    processor = VideoProcessor(
        task_id=task_id,
        upload_dir=os.path.join(UPLOAD_DIR, user_id),
        results_dir=os.path.join(RESULTS_DIR, user_id),
    )

    local_path = file_path

    try:
        if url and not local_path:
            local_path = await processor.download_from_url(url, task_id)
        
        if not local_path:
            raise ValueError("Не предоставлен файл или URL")

        await processing_lock.acquire()
        try:
            result = await processor.process_media(
                local_path, task_id, media_type, summary_type
            )
        finally:
            processing_lock.release()

        task_manager.complete_task(task_id, result)
        logger.info(f"Задача {task_id} завершена успешно")

    except Exception as e:
        logger.error(f"Ошибка обработки задачи {task_id}: {str(e)}", exc_info=True)
        task_manager.fail_task(task_id, str(e))

    finally:
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                logger.info(f"Временный файл удален: {local_path}")
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл {local_path}: {e}")


@app.get("/progress/{task_id}")
async def get_progress(
    task_id: str,
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
    client_id: Optional[str] = Query(default=None),
):
    """Получение прогресса по задаче."""
    user_id = _resolve_client_id(x_client_id=x_client_id, client_id=client_id)

    task_info = task_manager.get_task_info_for_user(user_id=user_id, task_id=task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": task_info["status"],
        "percent": task_info["percent"],
        "current_stage": task_info["current_stage"],
        "current_stage_display": task_info.get("current_stage_display", {}),
        "logs": task_info["logs"][-10:],
        "result": task_info.get("result"),
        "error": task_info.get("error"),
        "file_name": task_info.get("file_name"),
        "batch_id": task_info.get("batch_id"),
        "order_index": task_info.get("order_index"),
        "total_in_batch": task_info.get("total_in_batch"),
        "summary_type": task_info.get("summary_type"),
    }


@app.get("/batch/progress/{batch_id}")
async def get_batch_progress(
    batch_id: str,
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
    client_id: Optional[str] = Query(default=None),
):
    """Прогресс пакетной обработки (текущий/следующий файл + список задач)."""
    user_id = _resolve_client_id(x_client_id=x_client_id, client_id=client_id)

    batch_info = task_manager.get_batch_info_for_user(user_id=user_id, batch_id=batch_id)
    if not batch_info:
        raise HTTPException(status_code=404, detail="Batch not found")

    return batch_info


@app.get("/download/{task_id}/{file_type}")
async def download_file(
    task_id: str,
    file_type: str,
    x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
    client_id: Optional[str] = Query(default=None),
):
    """Скачивание результатов (из директории пользователя)."""
    user_id = _resolve_client_id(x_client_id=x_client_id, client_id=client_id)

    logger.info(
        f"Запрос на скачивание: task_id={task_id}, file_type={file_type}, user_id={user_id}"
    )

    if file_type == "summary":
        filename = f"{task_id}_summary.md"
    elif file_type == "transcription":
        filename = f"{task_id}_transcription.md"
    else:
        logger.error(f"Неизвестный тип файла: {file_type}")
        raise HTTPException(status_code=404, detail="Тип файла не поддерживается")

    file_path = os.path.join(RESULTS_DIR, user_id, filename)

    if not os.path.exists(file_path):
        logger.error(f"Файл не найден: {file_path}")
        raise HTTPException(status_code=404, detail="Файл не найден")

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
