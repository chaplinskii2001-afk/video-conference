"""
Кастомные исключения для учебного приложения обработки видео-конференций.
Адаптированы для образовательных целей с подробными объяснениями.
"""
from typing import Optional, Any


class VideoProcessorError(Exception):
    """Базовый класс исключений для обработки видео в учебной среде"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None, learning_tip: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.learning_tip = learning_tip  # Совет для обучения


class ConfigurationError(VideoProcessorError):
    """Ошибки конфигурации приложения в учебной среде"""
    def __init__(self, message: str, config_file: Optional[str] = None, learning_tip: Optional[str] = None):
        if learning_tip is None:
            learning_tip = "Проверьте файл конфигурации на наличие опечаток и правильных путей к моделям."
        super().__init__(message, details={"config_file": config_file}, learning_tip=learning_tip)


class ModelLoadingError(VideoProcessorError):
    """Ошибки загрузки AI моделей в учебной среде"""
    def __init__(self, model_name: str, reason: Optional[str] = None, learning_tip: Optional[str] = None):
        base_message = f"Ошибка загрузки модели {model_name}"
        if reason:
            base_message += f": {reason}"
        
        if learning_tip is None:
            learning_tips = {
                "whisper": "Убедитесь, что модель Whisper скачана и доступна. Проверьте HF_TOKEN в переменных окружения.",
                "diarization": "Проверьте, что PyAnnote модель загружена и вы приняли условия использования на HuggingFace.",
                "qwen": "Убедитесь, что модель Qwen скачана в правильную папку модели (/app/models/qwen)."
            }
            learning_tip = learning_tips.get(model_name.lower(), "Проверьте наличие модели и правильность путей.")
        
        super().__init__(base_message, details={"model_name": model_name, "reason": reason}, learning_tip=learning_tip)


class AudioProcessingError(VideoProcessorError):
    """Ошибки обработки аудио в учебной среде"""
    def __init__(self, operation: str, audio_path: str, details: Optional[str] = None, learning_tip: Optional[str] = None):
        base_message = f"Ошибка {operation} аудио файла: {audio_path}"
        if details:
            base_message += f" ({details})"
        
        if learning_tip is None:
            learning_tip = "Убедитесь, что ffmpeg установлен и файл не поврежден. Проверьте формат аудио."
        
        super().__init__(base_message, details={"audio_path": audio_path, "operation": operation}, learning_tip=learning_tip)


class TranscriptionError(VideoProcessorError):
    """Ошибки транскрипции в учебной среде"""
    def __init__(self, audio_path: str, reason: Optional[str] = None, learning_tip: Optional[str] = None):
        base_message = f"Ошибка транскрипции аудио {audio_path}"
        if reason:
            base_message += f": {reason}"
        
        if learning_tip is None:
            learning_tip = "Проверьте качество аудио, наличие речи и правильность формата. Убедитесь, что модель Whisper загружена."
        
        super().__init__(base_message, details={"audio_path": audio_path, "reason": reason}, learning_tip=learning_tip)


class DiarizationError(VideoProcessorError):
    """Ошибки диаризации в учебной среде"""
    def __init__(self, audio_path: str, reason: Optional[str] = None, learning_tip: Optional[str] = None):
        base_message = f"Ошибка диаризации аудио {audio_path}"
        if reason:
            base_message += f": {reason}"
        
        if learning_tip is None:
            learning_tip = "Диаризация работает лучше с качественным аудио. Убедитесь, что модель PyAnnote загружена и имеет доступ к GPU."
        
        super().__init__(base_message, details={"audio_path": audio_path, "reason": reason}, learning_tip=learning_tip)


class SummarizationError(VideoProcessorError):
    """Ошибки суммаризации в учебной среде"""
    def __init__(self, text_length: int, summary_type: str, reason: Optional[str] = None, learning_tip: Optional[str] = None):
        base_message = f"Ошибка суммаризации типа '{summary_type}' для текста длиной {text_length} символов"
        if reason:
            base_message += f": {reason}"
        
        if learning_tip is None:
            learning_tip = "Проверьте качество транскрипции и убедитесь, что модель Qwen загружена. Возможно, текст слишком короткий."
        
        super().__init__(base_message, details={"text_length": text_length, "summary_type": summary_type, "reason": reason}, learning_tip=learning_tip)


class FileProcessingError(VideoProcessorError):
    """Ошибки обработки файлов в учебной среде"""
    def __init__(self, file_path: str, operation: str, details: Optional[str] = None, learning_tip: Optional[str] = None):
        base_message = f"Ошибка {operation} файла {file_path}"
        if details:
            base_message += f" ({details})"
        
        if learning_tip is None:
            learning_tip = "Проверьте существование файла, права доступа и формат. Убедитесь, что ffmpeg установлен."
        
        super().__init__(base_message, details={"file_path": file_path, "operation": operation}, learning_tip=learning_tip)


class ValidationError(VideoProcessorError):
    """Ошибки валидации входных данных в учебной среде"""
    def __init__(self, parameter: str, value: Any, expected: str, learning_tip: Optional[str] = None):
        base_message = f"Неверное значение параметра '{parameter}': '{value}'. Ожидается: {expected}"
        
        if learning_tip is None:
            learning_tip = "Проверьте документацию по поддерживаемым форматам и ограничениям размера файлов."
        
        super().__init__(base_message, details={"parameter": parameter, "value": value, "expected": expected}, learning_tip=learning_tip)


class MemoryManagementError(VideoProcessorError):
    """Ошибки управления памятью в учебной среде"""
    def __init__(self, operation: str, details: Optional[str] = None, learning_tip: Optional[str] = None):
        base_message = f"Ошибка управления памятью при операции '{operation}'"
        if details:
            base_message += f": {details}"
        
        if learning_tip is None:
            learning_tip = "Попробуйте обрабатывать меньшие файлы или используйте модели с более агрессивным квантованием."
        
        super().__init__(base_message, details={"operation": operation}, learning_tip=learning_tip)


class TaskManagementError(VideoProcessorError):
    """Ошибки управления задачами в учебной среде"""
    def __init__(self, task_id: str, operation: str, details: Optional[str] = None, learning_tip: Optional[str] = None):
        base_message = f"Ошибка {operation} задачи {task_id}"
        if details:
            base_message += f": {details}"
        
        if learning_tip is None:
            learning_tip = "Проверьте, что задача существует и не превысила время ожидания. Посмотрите логи для подробностей."
        
        super().__init__(base_message, details={"task_id": task_id, "operation": operation}, learning_tip=learning_tip)


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
    Преобразует исключение в стандартизированный ответ для учебной среды
    
    Args:
        exception: Исходное исключение
        
    Returns:
        Словарь с информацией об ошибке и советами для обучения
    """
    if isinstance(exception, VideoProcessorError):
        response = {
            "error_type": exception.error_code,
            "message": exception.message,
            "details": exception.details,
            "educational_context": True,
        }
        
        # Добавляем совет для обучения если он есть
        if hasattr(exception, 'learning_tip') and exception.learning_tip:
            response["learning_tip"] = exception.learning_tip
            
        # Добавляем дополнительные образовательные подсказки
        if exception.error_code == "MODEL_LOADING_ERROR":
            response["debugging_suggestions"] = [
                "Проверьте переменные окружения (HF_TOKEN)",
                "Убедитесь, что модели скачаны в правильные папки",
                "Проверьте доступность интернета для загрузки моделей",
                "Посмотрите логи загрузки модели для подробностей"
            ]
        elif exception.error_code == "AUDIO_PROCESSING_ERROR":
            response["debugging_suggestions"] = [
                "Проверьте, что ffmpeg установлен: ffmpeg -version",
                "Убедитесь, что файл не поврежден",
                "Попробуйте другой формат аудио",
                "Проверьте права доступа к файлу"
            ]
        elif exception.error_code == "TRANSCRIPTION_ERROR":
            response["debugging_suggestions"] = [
                "Проверьте качество аудио (должна быть четкая речь)",
                "Убедитесь, что аудио содержит речь, а не музыку",
                "Проверьте, что Whisper модель загружена",
                "Попробуйте обрезать длинный файл на короткий участок"
            ]
        elif exception.error_code == "MEMORY_MANAGEMENT_ERROR":
            response["debugging_suggestions"] = [
                "Уменьшите размер обрабатываемого файла",
                "Используйте более агрессивное квантование моделей",
                "Проверьте доступную память GPU",
                "Закройте другие программы, использующие GPU"
            ]
        
        return response
    else:
        # Обработка стандартных исключений с образовательными подсказками
        error_mapping = {
            FileNotFoundError: {
                "code": "FILE_NOT_FOUND", 
                "tips": "Проверьте путь к файлу и его существование. Убедитесь, что файл не был удален."
            },
            PermissionError: {
                "code": "PERMISSION_DENIED", 
                "tips": "Проверьте права доступа к файлу или папке. Возможно, нужно запустить с другими правами."
            },
            ConnectionError: {
                "code": "CONNECTION_ERROR", 
                "tips": "Проверьте подключение к интернету для загрузки моделей с HuggingFace."
            },
            TimeoutError: {
                "code": "TIMEOUT_ERROR", 
                "tips": "Операция заняла слишком много времени. Попробуйте файл меньшего размера или более быстрый интернет."
            },
            ValueError: {
                "code": "INVALID_VALUE", 
                "tips": "Неверное значение параметра. Проверьте типы данных и формат входных параметров."
            },
            TypeError: {
                "code": "TYPE_ERROR", 
                "tips": "Неверный тип данных. Проверьте типы входных параметров функции."
            },
        }
        
        error_info = error_mapping.get(type(exception), {
            "code": "UNKNOWN_ERROR", 
            "tips": "Неизвестная ошибка. Посмотрите логи приложения для получения подробностей."
        })
        
        return {
            "error_type": error_info["code"],
            "message": str(exception),
            "details": {
                "exception_type": type(exception).__name__,
                "original_message": str(exception),
                "educational_tip": error_info["tips"]
            },
            "educational_context": True,
            "debugging_help": True
        }


def get_learning_resources(error_type: str) -> dict:
    """
    Предоставляет образовательные ресурсы для различных типов ошибок.
    
    Args:
        error_type: Тип ошибки
        
    Returns:
        Словарь с ресурсами для обучения
    """
    learning_resources = {
        "MODEL_LOADING_ERROR": {
            "concepts_to_study": [
                "Архитектура Transformer",
                "Квантование нейронных сетей", 
                "Управление памятью GPU"
            ],
            "documentation_links": [
                "https://huggingface.co/docs",
                "https://pytorch.org/docs/stable/cuda.html",
                "https://bitsandbytes.readthedocs.io"
            ],
            "practical_exercises": [
                "Попробуйте загрузить модель вручную через Python",
                "Экспериментируйте с разными режимами квантования",
                "Измерьте использование памяти GPU"
            ]
        },
        "AUDIO_PROCESSING_ERROR": {
            "concepts_to_study": [
                "Форматы аудио файлов",
                "Обработка аудио с помощью ffmpeg",
                "Частота дискретизации и каналы аудио"
            ],
            "documentation_links": [
                "https://ffmpeg.org/documentation.html",
                "https://docs.python.org/3/library/subprocess.html"
            ],
            "practical_exercises": [
                "Конвертируйте аудио файлы через ffmpeg вручную",
                "Изучите метаданные аудио файлов",
                "Сравните разные форматы аудио"
            ]
        },
        "TRANSCRIPTION_ERROR": {
            "concepts_to_study": [
                "Автоматическое распознавание речи (ASR)",
                "Whisper архитектура",
                "Качество аудио и его влияние на точность"
            ],
            "documentation_links": [
                "https://github.com/openai/whisper",
                "https://huggingface.co/docs/transformers/tasks/asr"
            ],
            "practical_exercises": [
                "Попробуйте транскрипцию с разным качеством аудио",
                "Измерьте точность на коротких и длинных записях",
                "Сравните модели разных размеров"
            ]
        },
        "MEMORY_MANAGEMENT_ERROR": {
            "concepts_to_study": [
                "Управление памятью в PyTorch",
                "Garbage Collection в Python",
                "Оптимизация использования GPU"
            ],
            "documentation_links": [
                "https://pytorch.org/docs/stable/cuda.html#memory-management",
                "https://docs.python.org/3/library/gc.html"
            ],
            "practical_exercises": [
                "Мониторьте использование памяти во время обработки",
                "Экспериментируйте с разными стратегиями очистки",
                "Оптимизируйте размеры батчей"
            ]
        }
    }
    
    return learning_resources.get(error_type, {
        "concepts_to_study": [
            "Основы отладки Python приложений",
            "Логирование и мониторинг",
            "Работа с исключениями"
        ],
        "documentation_links": [
            "https://docs.python.org/3/tutorial/errors.html",
            "https://docs.python.org/3/library/logging.html"
        ],
        "practical_exercises": [
            "Изучите стек-трейс исключения",
            "Добавьте дополнительное логирование",
            "Создайте простые тестовые случаи"
        ]
    })


def create_educational_error_response(exception: Exception) -> dict:
    """
    Создает образовательный ответ об ошибке с дополнительными ресурсами.
    
    Args:
        exception: Исходное исключение
        
    Returns:
        Расширенный ответ об ошибке для образовательных целей
    """
    base_response = handle_exception(exception)
    error_type = base_response.get("error_type", "UNKNOWN_ERROR")
    
    # Добавляем образовательные ресурсы
    base_response["learning_resources"] = get_learning_resources(error_type)
    
    # Добавляем информацию о контексте ошибки
    base_response["error_context"] = {
        "is_educational_project": True,
        "project_type": "academic_research",
        "version": "1.0.0",
        "support_info": "Это учебный проект. Обратитесь к преподавателю или документации проекта."
    }
    
    return base_response