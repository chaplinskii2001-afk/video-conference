"""
Type hints для приложения обработки видео-конференций.
Централизация всех типов для улучшения читаемости и поддержки кода.
"""
from typing import List, Dict, Optional, Union, Any, Literal, Callable, AsyncGenerator
from datetime import datetime
from enum import Enum


# ================ БАЗОВЫЕ ТИПЫ ================

MediaType = Literal["video", "audio"]
SummaryType = Literal["standard", "protocol"]
TaskStatus = Literal["created", "processing", "completed", "error"]

# ================ AI МОДЕЛИ ================

class ModelStatus(Enum):
    """Статус загрузки модели"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    UNLOADING = "unloading"


class WhisperSegment(Dict):
    """Сегмент транскрипции Whisper"""
    start: float
    end: Optional[float]
    text: str


class DiarizationSegment(Dict):
    """Сегмент диаризации PyAnnote"""
    start: float
    end: float
    speaker: str


class MergedSegment(Dict):
    """Объединенный сегмент с информацией о спикере"""
    start: float
    end: float
    speaker: str
    text: str


# ================ УПРАВЛЕНИЕ ЗАДАЧАМИ ================

class TaskInfo(Dict):
    """Информация о задаче"""
    status: TaskStatus
    percent: int
    current_stage: str
    logs: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error: Optional[str]


class ProcessingResult(Dict):
    """Результат обработки медиа"""
    task_id: str
    summary: str
    transcription_length: int
    segments_count: int
    speakers_count: int
    processing_time_minutes: float
    media_type: MediaType
    summary_type: SummaryType
    # Дополнительные поля для будущих расширений
    gpu_memory_used_mb: Optional[float] = None
    model_versions: Optional[Dict[str, str]] = None


# ================ МОНИТОРИНГ ================

class GPUMemoryInfo(Dict):
    """Информация о памяти GPU"""
    total_gb: float
    used_gb: float
    free_gb: float
    usage_percent: float


class SystemMemoryInfo(Dict):
    """Информация о системной памяти"""
    total_gb: float
    available_gb: float
    used_gb: float
    usage_percent: float


class ModelInfo(Dict):
    """Информация о модели"""
    model_id: str
    loaded: bool
    device: str
    quantization: Optional[str] = None
    version: Optional[str] = None
    size_mb: Optional[float] = None


class SystemHealth(Dict):
    """Состояние системы"""
    gpu_available: bool
    gpu_memory: GPUMemoryInfo
    system_memory: SystemMemoryInfo
    models_loaded: Dict[str, bool]
    uptime_seconds: Optional[int] = None


# ================ ФАЙЛЫ И ВАЛИДАЦИЯ ================

class FileValidationResult(Dict):
    """Результат валидации файла"""
    valid: bool
    file_size: Optional[int] = None
    file_extension: Optional[str] = None
    media_type: Optional[MediaType] = None
    error_message: Optional[str] = None


class MediaMetadata(Dict):
    """Метаданные медиа файла"""
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    file_size_mb: Optional[float] = None
    codec: Optional[str] = None
    bitrate: Optional[int] = None


# ================ API RESPONSE ТИПЫ ================

class APIResponse(Dict):
    """Базовый ответ API"""
    success: bool
    message: Optional[str] = None


class ProcessingResponse(APIResponse):
    """Ответ при запуске обработки"""
    task_id: str
    status: Literal["processing"]
    summary_type: SummaryType
    media_type: MediaType


class ProgressResponse(Dict):
    """Ответ с прогрессом задачи"""
    task_id: str
    status: TaskStatus
    percent: int
    current_stage: str
    logs: List[str]
    result: Optional[ProcessingResult]
    error: Optional[str]


class DownloadResponse(Dict):
    """Ответ для скачивания файла"""
    file_path: str
    media_type: str
    filename: str


class HealthResponse(Dict):
    """Ответ проверки здоровья"""
    status: Literal["healthy", "unhealthy"]
    gpu_available: bool
    directories: Dict[str, bool]
    server_time_tomsk: str


# ================ КОНФИГУРАЦИЯ ================

class ModelConfig(Dict):
    """Конфигурация AI моделей"""
    whisper_model_id: str
    diarization_model_id: str
    qwen_model_path: str
    
    # Whisper settings
    whisper_chunk_length: int
    whisper_stride_length: tuple
    whisper_quantization: str
    
    # PyAnnote settings
    diarization_batch_size: int
    audio_sample_rate: int
    audio_channels: int
    
    # Qwen settings
    qwen_quantization: str
    qwen_max_new_tokens: int
    qwen_repetition_penalty: float


class SecurityConfig(Dict):
    """Конфигурация безопасности"""
    cors_origins: List[str]
    max_file_size_mb: int
    max_file_size_bytes: int
    allowed_extensions: set
    allowed_video_extensions: set
    allowed_audio_extensions: set


class ProcessingConfig(Dict):
    """Конфигурация обработки"""
    gpu_memory_threshold: float
    cleanup_interval_seconds: int
    max_chunk_size: int
    max_summary_length: int
    log_retention_count: int
    task_timeout_minutes: int


# ================ СЕРВИСЫ ================

class WhisperServiceInterface:
    """Интерфейс сервиса Whisper"""
    async def initialize(self) -> None: ...
    async def transcribe(self, audio_path: str) -> List[WhisperSegment]: ...
    async def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def get_model_info(self) -> ModelInfo: ...


class DiarizationServiceInterface:
    """Интерфейс сервиса диаризации"""
    async def initialize(self) -> None: ...
    async def diarize(self, audio_path: str) -> List[DiarizationSegment]: ...
    async def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def get_model_info(self) -> ModelInfo: ...


class SummarizationServiceInterface:
    """Интерфейс сервиса суммаризации"""
    async def initialize(self) -> None: ...
    async def summarize(
        self, 
        text: str, 
        summary_type: SummaryType, 
        max_length: int
    ) -> str: ...
    async def smart_summarize_long_text(
        self, 
        text: str, 
        summary_type: SummaryType
    ) -> str: ...
    async def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def get_model_info(self) -> ModelInfo: ...


# ================ CALLBACKS И EVENTS ================

class ProgressCallback(Callable[[int, str, str], None]):
    """Callback для обновления прогресса"""
    def __call__(self, percent: int, stage: str, message: str) -> None: ...


class LogCallback(Callable[[str], None]):
    """Callback для логирования"""
    def __call__(self, message: str) -> None: ...


class ErrorCallback(Callable[[Exception], None]):
    """Callback для обработки ошибок"""
    def __call__(self, error: Exception) -> None: ...


# ================ УТИЛИТЫ ================

class TimeRange(Dict):
    """Временной диапазон"""
    start: float
    end: float


class SpeakerInfo(Dict):
    """Информация о спикере"""
    speaker_id: str
    segments_count: int
    total_duration: float
    first_appearance: float
    last_appearance: float


class ProcessingStage(Dict):
    """Этап обработки"""
    name: str
    description: str
    percent_start: int
    percent_end: int
    optional: bool = False


# ================ ПРОМЕТЕУС МЕТРИКИ ================

class MetricsLabels(Dict):
    """Лейблы для метрик Prometheus"""
    model_name: str
    operation: str
    status: str
    gpu_id: Optional[str] = None
    media_type: Optional[MediaType] = None


# Экспортируем основные типы для удобства использования
__all__ = [
    # Базовые типы
    "MediaType", "SummaryType", "TaskStatus",
    
    # Модели данных
    "WhisperSegment", "DiarizationSegment", "MergedSegment",
    "TaskInfo", "ProcessingResult", "GPUMemoryInfo", "SystemMemoryInfo",
    "ModelInfo", "SystemHealth", "FileValidationResult", "MediaMetadata",
    
    # API типы
    "APIResponse", "ProcessingResponse", "ProgressResponse", 
    "DownloadResponse", "HealthResponse",
    
    # Конфигурация
    "ModelConfig", "SecurityConfig", "ProcessingConfig",
    
    # Интерфейсы сервисов
    "WhisperServiceInterface", "DiarizationServiceInterface", 
    "SummarizationServiceInterface",
    
    # Callbacks
    "ProgressCallback", "LogCallback", "ErrorCallback",
    
    # Утилиты
    "TimeRange", "SpeakerInfo", "ProcessingStage", "MetricsLabels",
]