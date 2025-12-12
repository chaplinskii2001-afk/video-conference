"""
Кастомные исключения для приложения обработки видео-конференций.
Обеспечивают четкую типизацию ошибок и лучшую обработку исключений.
"""
from typing import Optional, Any


class VideoProcessorError(Exception):
    """Базовый класс исключений для обработки видео"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class ConfigurationError(VideoProcessorError):
    """Ошибки конфигурации приложения"""
    pass


class ModelLoadingError(VideoProcessorError):
    """Ошибки загрузки AI моделей"""
    pass


class AudioProcessingError(VideoProcessorError):
    """Ошибки обработки аудио"""
    pass


class TranscriptionError(VideoProcessorError):
    """Ошибки транскрипции"""
    pass


class DiarizationError(VideoProcessorError):
    """Ошибки диаризации"""
    pass


class SummarizationError(VideoProcessorError):
    """Ошибки суммаризации"""
    pass


class FileProcessingError(VideoProcessorError):
    """Ошибки обработки файлов"""
    pass


class ValidationError(VideoProcessorError):
    """Ошибки валидации входных данных"""
    pass


class MemoryManagementError(VideoProcessorError):
    """Ошибки управления памятью"""
    pass


class TaskManagementError(VideoProcessorError):
    """Ошибки управления задачами"""
    pass


# Специализированные исключения для конкретных случаев


class UnsupportedFileFormatError(FileProcessingError):
    """Неподдерживаемый формат файла"""
    def __init__(self, file_format: str, supported_formats: list):
        message = f"Неподдерживаемый формат файла: {file_format}. Поддерживаемые: {', '.join(supported_formats)}"
        super().__init__(message, details={"file_format": file_format, "supported_formats": supported_formats})


class FileSizeExceededError(FileProcessingError):
    """Превышен максимальный размер файла"""
    def __init__(self, file_size: int, max_size: int):
        message = f"Размер файла {file_size} байт превышает максимальный {max_size} байт"
        super().__init__(message, details={"file_size": file_size, "max_size": max_size})


class ModelNotFoundError(ModelLoadingError):
    """Модель не найдена"""
    def __init__(self, model_path: str, model_type: str):
        message = f"Модель {model_type} не найдена по пути: {model_path}"
        super().__init__(message, details={"model_path": model_path, "model_type": model_type})


class GPUUnavailableError(ModelLoadingError):
    """GPU недоступна"""
    def __init__(self, reason: Optional[str] = None):
        message = f"GPU недоступна{' - ' + reason if reason else ''}"
        super().__init__(message, details={"reason": reason})


class AudioExtractionError(AudioProcessingError):
    """Ошибка извлечения аудио"""
    def __init__(self, video_path: str, details: Optional[str] = None):
        message = f"Ошибка извлечения аудио из {video_path}"
        if details:
            message += f": {details}"
        super().__init__(message, details={"video_path": video_path, "extraction_details": details})


class AudioConversionError(AudioProcessingError):
    """Ошибка конвертации аудио"""
    def __init__(self, audio_path: str, target_format: str):
        message = f"Ошибка конвертации аудио {audio_path} в формат {target_format}"
        super().__init__(message, details={"audio_path": audio_path, "target_format": target_format})


class TranscriptionTimeoutError(TranscriptionError):
    """Превышено время ожидания транскрипции"""
    def __init__(self, audio_duration: float, timeout_seconds: int):
        message = f"Транскрипция превысила таймаут {timeout_seconds}s для аудио длительностью {audio_duration}s"
        super().__init__(message, details={"audio_duration": audio_duration, "timeout_seconds": timeout_seconds})


class DiarizationFailedError(DiarizationError):
    """Не удалось выполнить диаризацию"""
    def __init__(self, audio_path: str, reason: Optional[str] = None):
        message = f"Диаризация не удалась для {audio_path}"
        if reason:
            message += f": {reason}"
        super().__init__(message, details={"audio_path": audio_path, "failure_reason": reason})


class SummarizationFailedError(SummarizationError):
    """Ошибка суммаризации"""
    def __init__(self, text_length: int, summary_type: str, reason: Optional[str] = None):
        message = f"Суммаризация типа '{summary_type}' не удалась для текста длиной {text_length} символов"
        if reason:
            message += f": {reason}"
        super().__init__(message, details={"text_length": text_length, "summary_type": summary_type, "reason": reason})


class TaskNotFoundError(TaskManagementError):
    """Задача не найдена"""
    def __init__(self, task_id: str):
        message = f"Задача с ID {task_id} не найдена"
        super().__init__(message, details={"task_id": task_id})


class TaskTimeoutError(TaskManagementError):
    """Превышено время выполнения задачи"""
    def __init__(self, task_id: str, timeout_minutes: int):
        message = f"Задача {task_id} превысила таймаут {timeout_minutes} минут"
        super().__init__(message, details={"task_id": task_id, "timeout_minutes": timeout_minutes})


class InvalidURLError(ValidationError):
    """Недействительный URL"""
    def __init__(self, url: str, reason: Optional[str] = None):
        message = f"Недействительный URL: {url}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, details={"url": url, "reason": reason})


class EmptyTranscriptionError(TranscriptionError):
    """Транскрипция пуста"""
    def __init__(self, audio_path: str):
        message = f"Транскрипция для аудио {audio_path} не дала результатов"
        super().__init__(message, details={"audio_path": audio_path})


class InsufficientGPUMemoryError(MemoryManagementError):
    """Недостаточно памяти GPU"""
    def __init__(self, required_gb: float, available_gb: float):
        message = f"Недостаточно GPU памяти: требуется {required_gb}GB, доступно {available_gb}GB"
        super().__init__(message, details={"required_gb": required_gb, "available_gb": available_gb})


class CorruptedMediaFileError(FileProcessingError):
    """Поврежденный медиа файл"""
    def __init__(self, file_path: str, corruption_details: Optional[str] = None):
        message = f"Медиа файл {file_path} поврежден"
        if corruption_details:
            message += f": {corruption_details}"
        super().__init__(message, details={"file_path": file_path, "corruption_details": corruption_details})


# Функции-помощники для обработки исключений


def handle_exception(exception: Exception) -> dict:
    """
    Преобразует исключение в стандартизированный ответ
    
    Args:
        exception: Исходное исключение
        
    Returns:
        Словарь с информацией об ошибке
    """
    if isinstance(exception, VideoProcessorError):
        return {
            "error_type": exception.error_code,
            "message": exception.message,
            "details": exception.details,
        }
    else:
        # Обработка стандартных исключений
        error_mapping = {
            FileNotFoundError: "FILE_NOT_FOUND",
            PermissionError: "PERMISSION_DENIED", 
            ConnectionError: "CONNECTION_ERROR",
            TimeoutError: "TIMEOUT_ERROR",
            ValueError: "INVALID_VALUE",
            TypeError: "TYPE_ERROR",
        }
        
        error_type = error_mapping.get(type(exception), "UNKNOWN_ERROR")
        
        return {
            "error_type": error_type,
            "message": str(exception),
            "details": {
                "exception_type": type(exception).__name__,
                "original_message": str(exception),
            }
        }