"""
Конфигурация приложения.
Централизованное управление настройками для безопасности и производительности.
"""
import os
from typing import List, Optional
from functools import lru_cache


class Settings:
    """Настройки приложения с валидацией"""
    
    # ================= БЕЗОПАСНОСТЬ =================
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:8000"
    ]
    
    # Ограничения загрузки файлов
    MAX_FILE_SIZE_MB: int = 500
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Поддерживаемые форматы файлов
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
    ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}
    ALLOWED_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
    
    # ================= AI МОДЕЛИ =================
    WHISPER_MODEL_ID: str = "bond005/whisper-podlodka-turbo"
    DIARIZATION_MODEL_ID: str = "pyannote/speaker-diarization-3.1"
    
    # Пути к моделям
    WHISPER_MODEL_PATH: str = "/app/models/whisper"
    PYANNOTE_CACHE_PATH: str = "/app/models/pyannote" 
    QWEN_MODEL_PATH: str = "/app/models/qwen"
    
    # ================= ПРОИЗВОДИТЕЛЬНОСТЬ =================
    # Управление памятью
    GPU_MEMORY_THRESHOLD: float = 0.85  # 85% максимум
    CLEANUP_INTERVAL_SECONDS: int = 300  # 5 минут
    
    # Обработка текста
    MAX_CHUNK_SIZE: int = 14000
    MAX_SUMMARY_LENGTH: int = 4000
    LOG_RETENTION_COUNT: int = 50
    
    # ================= ЗАДАЧИ =================
    TASK_CLEANUP_INTERVAL_HOURS: int = 1
    TASK_TIMEOUT_MINUTES: int = 120  # 2 часа максимум
    
    # ================= ИНФРАСТРУКТУРА =================
    # Директории
    UPLOAD_DIR: str = "uploads"
    RESULTS_DIR: str = "results"
    LOGS_DIR: str = "/app/logs"
    
    # Медиа конвертация
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1  # Моно
    
    # ================= ВРЕМЯ =================
    # Томск timezone (UTC+7)
    TOMSK_TIMEZONE_OFFSET: int = 7


class SecurityConfig:
    """Безопасность и валидация"""
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Валидация расширения файла"""
        if not filename:
            return False
        ext = os.path.splitext(filename)[1].lower()
        return ext in Settings.ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Валидация размера файла"""
        return file_size <= Settings.MAX_FILE_SIZE_BYTES
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Безопасное имя файла"""
        if not filename:
            return "unnamed_file"
        
        # Удаляем небезопасные символы
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        sanitized = "".join(c for c in filename if c in safe_chars)
        
        # Ограничиваем длину
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:200-len(ext)] + ext
            
        return sanitized or "unnamed_file"


class ModelConfig:
    """Конфигурация AI моделей"""
    
    # Whisper settings
    WHISPER_CHUNK_LENGTH: int = 30
    WHISPER_STRIDE_LENGTH: tuple = (4, 2)
    
    # Qwen settings  
    QWEN_MAX_NEW_TOKENS: int = 800
    QWEN_REPETITION_PENALTY: float = 1.05
    
    # PyAnnote settings
    DIARIZATION_BATCH_SIZE: int = 16


@lru_cache()
def get_settings() -> Settings:
    """Получение настроек с кэшированием"""
    return Settings()


def get_security_config() -> SecurityConfig:
    """Получение конфигурации безопасности"""
    return SecurityConfig()


def get_model_config() -> ModelConfig:
    """Получение конфигурации моделей"""
    return ModelConfig()