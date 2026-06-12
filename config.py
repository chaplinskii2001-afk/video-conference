"""
Конфигурация приложения для масштабируемости
Этот файл содержит настройки, которые легко адаптируются под разное железо
"""
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GPUConfig:
    """Конфигурация на основе доступной GPU памяти"""
    
    # Определяем профили для разных классов GPU
    PROFILES = {
        "low": {
            "name": "Базовый (4-6 GB VRAM)",
            "min_vram_gb": 0,
            "max_vram_gb": 6,
            "whisper_quantization": "float16",
            "qwen_quantization": "4bit",
            "batch_size": 2,
            "diarization_batch_size": 16,
            "max_audio_length_minutes": 120,
            "chunk_length_s": 30,
            "stride_length_s": (4, 2),
        },
        "medium": {
            "name": "Средний (8-12 GB VRAM)",
            "min_vram_gb": 6,
            "max_vram_gb": 12,
            "whisper_quantization": "float32",
            "qwen_quantization": "8bit",
            "batch_size": 2,
            "diarization_batch_size": 32,
            "max_audio_length_minutes": 140,
            "chunk_length_s": 30,
            "stride_length_s": (5, 2),
        },
        "high": {
            "name": "Мощный (16-24 GB VRAM)",
            "min_vram_gb": 12,
            "max_vram_gb": 24,
            "whisper_quantization": "float32",
            "qwen_quantization": "float16",
            "batch_size": 4,
            "diarization_batch_size": 64,
            "max_audio_length_minutes": 180,
            "chunk_length_s": 30,
            "stride_length_s": (6, 2),
        },
        "ultra": {
            "name": "Профессиональный (24+ GB VRAM)",
            "min_vram_gb": 24,
            "max_vram_gb": 999,
            "whisper_quantization": "float32",
            "qwen_quantization": "float16",
            "batch_size": 8,
            "diarization_batch_size": 128,
            "max_audio_length_minutes": 300,
            "chunk_length_s": 30,
            "stride_length_s": (6, 2),
        }
    }

    @staticmethod
    def detect_gpu_info() -> Dict[str, Any]:
        """Определяет информацию о GPU"""
        if not torch.cuda.is_available():
            return {
                "available": False,
                "vram_gb": 0,
                "name": "CPU",
                "profile": "cpu"
            }
        
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        
        return {
            "available": True,
            "vram_gb": round(vram_gb, 2),
            "name": gpu_name,
            "profile": GPUConfig._get_profile_for_vram(vram_gb)
        }
    
    @staticmethod
    def _get_profile_for_vram(vram_gb: float) -> str:
        """Определяет профиль на основе доступной VRAM"""
        for profile_name, profile_config in GPUConfig.PROFILES.items():
            if profile_config["min_vram_gb"] <= vram_gb < profile_config["max_vram_gb"]:
                return profile_name
        return "low"
    
    @staticmethod
    def get_config_for_current_gpu() -> Dict[str, Any]:
        """Возвращает оптимальную конфигурацию для текущего GPU"""
        gpu_info = GPUConfig.detect_gpu_info()
        
        if not gpu_info["available"]:
            logger.warning("GPU недоступен, используется CPU режим")
            return GPUConfig._get_cpu_config()
        
        profile = gpu_info["profile"]
        config = GPUConfig.PROFILES[profile].copy()
        config["gpu_info"] = gpu_info
        
        logger.info(f"Обнаружен GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB)")
        logger.info(f"Используется профиль: {config['name']}")
        
        return config
    
    @staticmethod
    def _get_cpu_config() -> Dict[str, Any]:
        """Конфигурация для CPU (резервный режим)"""
        return {
            "name": "CPU режим",
            "whisper_quantization": "8bit",
            "qwen_quantization": "4bit",
            "batch_size": 1,
            "diarization_batch_size": 8,
            "max_audio_length_minutes": 30,
            "chunk_length_s": 20,
            "stride_length_s": (3, 1),
            "gpu_info": {"available": False, "vram_gb": 0, "name": "CPU"}
        }


class AppConfig:
    """Общие настройки приложения"""
    
    # Пути к директориям
    UPLOAD_DIR = "uploads"
    RESULTS_DIR = "results"
    LOGS_DIR = "/app/logs"
    MODELS_DIR = "/app/models"
    
    # Пути к моделям
    WHISPER_MODEL_ID = "bond005/whisper-podlodka-turbo"
    WHISPER_MODEL_PATH = f"{MODELS_DIR}/whisper"
    
    DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
    DIARIZATION_MODEL_PATH = f"{MODELS_DIR}/pyannote"
    
    QWEN_MODEL_PATH = f"{MODELS_DIR}/qwen"
    
    # Поддерживаемые форматы
    VIDEO_FORMATS = [".mp4", ".webm", ".mov", ".avi", ".mkv"]
    AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"]
    
    # Лимиты
    MAX_FILE_SIZE_MB = 2048  # 2 GB
    TASK_CLEANUP_HOURS = 1
    MAX_CONCURRENT_TASKS = 3
    
    # Настройки логирования
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Настройки суммаризации
    SUMMARIZATION_MAX_CHARS = 14000
    SUMMARY_MAX_NEW_TOKENS = {
        "standard": 800,
        "protocol": 1200,
        "final": 7000
    }
    
    @staticmethod
    def get_gpu_config() -> Dict[str, Any]:
        """Получить конфигурацию GPU"""
        return GPUConfig.get_config_for_current_gpu()
    
    @staticmethod
    def get_model_config() -> Dict[str, str]:
        """Получить конфигурацию моделей"""
        return {
            "whisper_id": AppConfig.WHISPER_MODEL_ID,
            "whisper_path": AppConfig.WHISPER_MODEL_PATH,
            "diarization_id": AppConfig.DIARIZATION_MODEL_ID,
            "diarization_path": AppConfig.DIARIZATION_MODEL_PATH,
            "qwen_path": AppConfig.QWEN_MODEL_PATH,
        }
