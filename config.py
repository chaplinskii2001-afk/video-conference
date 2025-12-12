"""
Конфигурация приложения.
Централизованное управление настройками для безопасности и производительности.
"""
import os
from typing import List, Optional
from functools import lru_cache


class Settings:
    """Настройки приложения с валидацией"""
    
    # ================= БЕЗОПАСНОСТЬ ДЛЯ УЧЕБНОЙ СРЕДЫ =================
    # Для локальной сети института - более гибкие настройки
    CORS_ORIGINS: List[str] = [
        "http://localhost:*",          # Локальная разработка
        "http://127.0.0.1:*",         # Локальная разработка  
        "http://192.168.*",           # Локальная сеть
        "http://10.*",                # Локальная сеть
        "http://172.16.*",            # Локальная сеть
        "http://*.edu.ru",            # Домены институтов
        "http://*.ac.ru",             # Академические домены
    ]
    
    # Ограничения загрузки файлов (адаптировано для исследовательских задач)
    MAX_FILE_SIZE_MB: int = 1000  # Увеличено до 1GB для исследований
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
    
    # ================= МАСШТАБИРОВАНИЕ ДЛЯ ИССЛЕДОВАНИЙ =================
    # Настройки для будущего масштабирования на мощное железо
    GPU_MEMORY_THRESHOLD: float = 0.85  # 85% максимум (можно увеличить для мощных GPU)
    CLEANUP_INTERVAL_SECONDS: int = 300  # 5 минут
    
    # Настройки для разных уровней производительности
    PERFORMANCE_LEVEL: str = "balanced"  # "research", "balanced", "performance"
    
    # Режимы работы модели для разных GPU
    WHISPER_QUANTIZATION: str = "8bit"  # 8bit, 16bit, 32bit для разной производительности
    QWEN_QUANTIZATION: str = "4bit"     # 4bit, 8bit, 16bit для разной производительности
    
    # Настройки для расширения функциональности
    ENABLE_EXPERIMENTAL_FEATURES: bool = False  # Для будущих исследований
    BATCH_PROCESSING_ENABLED: bool = False     # Для пакетной обработки
    
    # ================= ВРЕМЯ =================
    # Томск timezone (UTC+7)
    TOMSK_TIMEZONE_OFFSET: int = 7


class SecurityConfig:
    """Безопасность и валидация для учебного проекта"""
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Валидация расширения файла"""
        if not filename:
            return False
        ext = os.path.splitext(filename)[1].lower()
        return ext in Settings.ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Валидация размера файла (адаптировано для исследовательских задач)"""
        return file_size <= Settings.MAX_FILE_SIZE_BYTES
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Безопасное имя файла"""
        if not filename:
            return "unnamed_file"
        
        # Для учебного проекта оставляем больше гибкости
        safe_chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        sanitized = "".join(c for c in filename if c in safe_chars)
        
        # Ограничиваем длину
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:200-len(ext)] + ext
            
        return sanitized or "unnamed_file"
    
    @staticmethod
    def is_educational_environment() -> bool:
        """Определяет, работает ли система в учебной среде"""
        return os.getenv("EDUCATIONAL_MODE", "true").lower() == "true"


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


def get_performance_config(gpu_memory_gb: float = 0) -> Dict[str, Any]:
    """
    Получение конфигурации производительности на основе доступного GPU.
    
    Args:
        gpu_memory_gb: Доступная память GPU в GB
        
    Returns:
        Словарь с настройками производительности
    """
    settings = get_settings()
    
    if gpu_memory_gb >= 24:  # Мощные GPU (A100, RTX 4090 и выше)
        return {
            "whisper_quantization": "16bit",  # Лучшее качество
            "qwen_quantization": "8bit",      # Лучшее качество для Qwen
            "max_concurrent_tasks": 4,        # Больше параллельных задач
            "chunk_size_multiplier": 2.0,     # Большие чанки для обработки
            "memory_threshold": 0.90,         # Более высокий порог памяти
        }
    elif gpu_memory_gb >= 12:  # Средние GPU (RTX 3080, 4070 Ti)
        return {
            "whisper_quantization": "8bit",   # Баланс качества/производительности
            "qwen_quantization": "4bit",      # Стандартная настройка
            "max_concurrent_tasks": 2,        # Умеренная параллельность
            "chunk_size_multiplier": 1.0,     # Стандартные чанки
            "memory_threshold": 0.85,         # Стандартный порог
        }
    elif gpu_memory_gb >= 8:   # Бюджетные GPU (RTX 3060, 3070)
        return {
            "whisper_quantization": "8bit",   # Для экономии памяти
            "qwen_quantization": "4bit",      # Минимальные требования
            "max_concurrent_tasks": 1,        # Последовательная обработка
            "chunk_size_multiplier": 0.8,     # Меньшие чанки
            "memory_threshold": 0.80,         # Более консервативный порог
        }
    else:  # Слабые GPU или CPU
        return {
            "whisper_quantization": "8bit",   # Только 8bit для экономии памяти
            "qwen_quantization": "4bit",      # Только 4bit
            "max_concurrent_tasks": 1,        # Только последовательно
            "chunk_size_multiplier": 0.5,     # Очень маленькие чанки
            "memory_threshold": 0.75,         # Очень консервативный порог
        }


def detect_gpu_capabilities() -> Dict[str, Any]:
    """
    Определение возможностей GPU для оптимальной настройки.
    
    Returns:
        Словарь с информацией о GPU
    """
    try:
        import torch
        import pynvml
        
        if not torch.cuda.is_available():
            return {
                "gpu_available": False,
                "memory_gb": 0,
                "compute_capability": "0.0",
                "recommended_settings": "cpu_only"
            }
        
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        memory_gb = memory_info.total / 1024**3
        
        # Получаем Compute Capability
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = f"{major}.{minor}"
        
        # Определяем рекомендуемые настройки
        perf_config = get_performance_config(memory_gb)
        
        return {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "memory_gb": round(memory_gb, 1),
            "memory_free_gb": round(memory_info.free / 1024**3, 1),
            "compute_capability": compute_capability,
            "recommended_settings": perf_config,
            "suitable_for_research": memory_gb >= 12,
        }
        
    except Exception as e:
        return {
            "gpu_available": False,
            "memory_gb": 0,
            "compute_capability": "unknown",
            "recommended_settings": "cpu_only",
            "error": str(e)
        }


def get_educational_setup() -> Dict[str, Any]:
    """
    Получение настроек, оптимальных для учебной/исследовательской среды.
    
    Returns:
        Словарь с учебными настройками
    """
    gpu_info = detect_gpu_capabilities()
    
    # Базовые настройки для учебной среды
    base_settings = {
        "max_file_size_mb": 1000,           # Большие файлы для исследований
        "max_processing_time_hours": 4,     # Длительное время для сложных задач
        "enable_experimental": True,        # Разрешаем экспериментальные функции
        "save_intermediate_results": True,  # Сохраняем промежуточные результаты
        "detailed_logging": True,           # Подробное логирование для анализа
        "multiple_language_support": True,  # Поддержка разных языков
    }
    
    # Дополнительные настройки в зависимости от GPU
    if gpu_info["gpu_available"]:
        research_settings = gpu_info["recommended_settings"]
        research_settings.update({
            "suitable_for_research": gpu_info["suitable_for_research"],
            "gpu_recommendations": _generate_gpu_recommendations(gpu_info),
        })
    else:
        research_settings = {
            "gpu_available": False,
            "cpu_only_mode": True,
            "suitable_for_research": False,
            "recommendation": "Рекомендуется использовать GPU для исследований",
        }
    
    return {
        "educational_mode": True,
        "base_settings": base_settings,
        "hardware_info": gpu_info,
        "research_settings": research_settings,
    }


def _generate_gpu_recommendations(gpu_info: Dict[str, Any]) -> Dict[str, Any]:
    """Генерация рекомендаций по использованию GPU"""
    memory_gb = gpu_info["memory_gb"]
    
    recommendations = []
    limitations = []
    
    if memory_gb >= 24:
        recommendations.extend([
            "Отлично подходит для исследований",
            "Можно использовать модели без квантования",
            "Поддерживает пакетную обработку",
            "Подходит для обучения моделей"
        ])
    elif memory_gb >= 12:
        recommendations.extend([
            "Хорошо подходит для исследований",
            "Рекомендуется 8-bit квантование для Whisper",
            "Подходит для большинства исследовательских задач"
        ])
    elif memory_gb >= 8:
        recommendations.extend([
            "Минимальные требования для комфортной работы",
            "Рекомендуется использовать все модели с квантованием",
            "Лучше обрабатывать файлы последовательно"
        ])
        limitations.extend([
            "Ограниченная поддержка пакетной обработки",
            "Не рекомендуется для обучения моделей"
        ])
    else:
        limitations.extend([
            "Рекомендуется использовать только для простых задач",
            "Возможны задержки при обработке длинных файлов",
            "Для серьезных исследований рекомендуется GPU с 8+ GB памяти"
        ])
    
    return {
        "recommendations": recommendations,
        "limitations": limitations,
        "optimal_workflow": _suggest_optimal_workflow(memory_gb)
    }


def _suggest_optimal_workflow(memory_gb: float) -> str:
    """Предложение оптимального рабочего процесса"""
    if memory_gb >= 16:
        return "Параллельная обработка нескольких файлов с полным качеством моделей"
    elif memory_gb >= 8:
        return "Последовательная обработка с оптимизированными моделями (8-bit/4-bit)"
    else:
        return "Обработка коротких файлов с максимальной оптимизацией памяти"