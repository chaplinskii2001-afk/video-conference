"""
Менеджер AI моделей
Управляет загрузкой, выгрузкой и переключением между моделями
"""
import os
import torch
import logging
from typing import Optional, Dict
from transformers import (
    pipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from pyannote.audio import Pipeline
from processing.gpu_manager import GPUMemoryManager


class ModelManager:
    """
    Менеджер моделей AI
    - Загружает модели по требованию
    - Выгружает неиспользуемые модели для экономии памяти
    - Адаптирует параметры загрузки под доступное GPU
    """
    
    def __init__(self, config: Dict, gpu_manager: GPUMemoryManager):
        self.config = config
        self.gpu_manager = gpu_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Токен HuggingFace
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Загруженные модели
        self.whisper_model = None
        self.whisper_processor = None
        self.whisper_pipeline = None
        
        self.diarization_pipeline = None
        
        self.qwen_model = None
        self.qwen_tokenizer = None
        
        self.current_loaded_model = None
        
        # Получаем конфигурацию GPU
        self.gpu_config = config.get("gpu_config", {})
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.whisper_batch_size = int(self.gpu_config.get("batch_size", 1) or 1)
        self.max_audio_length_minutes = self.gpu_config.get("max_audio_length_minutes")

        self.logger.info(f"ModelManager инициализирован для устройства: {self.device}")
        self.logger.info(
            "Параметры профиля: "
            f"batch_size={self.whisper_batch_size}, "
            f"max_audio_length_minutes={self.max_audio_length_minutes}"
        )
    
    # ==================== WHISPER ====================
    
    async def load_whisper(self) -> bool:
        """
        Загрузка модели Whisper для транскрипции
        Использует квантование на основе доступной GPU памяти
        """
        if self.whisper_pipeline is not None and self.current_loaded_model == "whisper":
            self.logger.info("Whisper уже загружен")
            return True
        
        # Выгружаем другие модели
        await self.unload_current_model()
        
        self.logger.info("Загрузка Whisper модели...")
        self.gpu_manager.take_snapshot("before_whisper")
        
        try:
            whisper_path = self.config["model_config"]["whisper_path"]
            whisper_id = self.config["model_config"]["whisper_id"]
            
            # Определяем тип квантования
            quantization = self.gpu_config.get("whisper_quantization", "8bit")
            
            self.logger.info(f"Whisper квантование: {quantization}")
            
            # Загружаем процессор
            self.whisper_processor = WhisperProcessor.from_pretrained(whisper_id)
            
            # Загружаем модель с квантованием
            if quantization == "8bit":
                model = WhisperForConditionalGeneration.from_pretrained(
                    whisper_id,
                    load_in_8bit=True,
                    device_map="auto",
                    torch_dtype=torch.float16 if self.device == "cuda:0" else torch.float32,
                )
            elif quantization == "float16":
                model = WhisperForConditionalGeneration.from_pretrained(
                    whisper_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            else:  # float32
                model = WhisperForConditionalGeneration.from_pretrained(
                    whisper_id,
                    device_map="auto",
                    torch_dtype=torch.float32,
                )
            
            # Создаем pipeline
            chunk_length = self.gpu_config.get("chunk_length_s", 30)
            stride_length = self.gpu_config.get("stride_length_s", (4, 2))
            batch_size = int(self.gpu_config.get("batch_size", 1) or 1)
            max_audio_length_minutes = self.gpu_config.get("max_audio_length_minutes")

            self.whisper_batch_size = batch_size
            self.max_audio_length_minutes = max_audio_length_minutes

            self.logger.info(
                "Whisper параметры: "
                f"chunk_length_s={chunk_length}, "
                f"stride_length_s={stride_length}, "
                f"batch_size={batch_size}, "
                f"max_audio_length_minutes={max_audio_length_minutes}"
            )

            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=self.whisper_processor.tokenizer,
                feature_extractor=self.whisper_processor.feature_extractor,
                chunk_length_s=chunk_length,
                stride_length_s=stride_length,
                return_timestamps=True,
            )
            
            self.current_loaded_model = "whisper"
            self.gpu_manager.take_snapshot("after_whisper")
            
            self.logger.info("✅ Whisper загружен успешно")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки Whisper: {e}", exc_info=True)
            await self.gpu_manager.cleanup("deep")
            return False
    
    # ==================== PYANNOTE (DIARIZATION) ====================
    
    async def load_diarization(self) -> bool:
        """
        Загрузка модели PyAnnote для диаризации спикеров
        """
        if self.diarization_pipeline is not None and self.current_loaded_model == "diarization":
            self.logger.info("PyAnnote уже загружен")
            return True
        
        # Выгружаем другие модели
        await self.unload_current_model()
        
        self.logger.info("Загрузка PyAnnote модели...")
        self.gpu_manager.take_snapshot("before_diarization")
        
        try:
            diarization_id = self.config["model_config"]["diarization_id"]
            diarization_path = self.config["model_config"]["diarization_path"]
            
            # Загружаем pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                diarization_id,
                token=self.hf_token,
                cache_dir=diarization_path,
            )
            
            # Переносим на GPU если доступно
            if torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
                self.logger.info("PyAnnote перенесен на GPU")
            
            self.current_loaded_model = "diarization"
            self.gpu_manager.take_snapshot("after_diarization")
            
            self.logger.info("✅ PyAnnote загружен успешно")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки PyAnnote: {e}", exc_info=True)
            self.logger.error("Проверьте HF_TOKEN и права доступа к модели")
            await self.gpu_manager.cleanup("deep")
            return False
    
    # ==================== QWEN (SUMMARIZATION) ====================
    
    async def load_qwen(self) -> bool:
        """
        Загрузка модели Qwen для суммаризации
        Использует квантование на основе доступной GPU памяти
        """
        if self.qwen_model is not None and self.current_loaded_model == "qwen":
            self.logger.info("Qwen уже загружен")
            return True
        
        # Выгружаем другие модели
        await self.unload_current_model()
        
        self.logger.info("Загрузка Qwen модели...")
        self.gpu_manager.take_snapshot("before_qwen")
        
        try:
            qwen_path = self.config["model_config"]["qwen_path"]
            
            if not os.path.exists(qwen_path):
                raise FileNotFoundError(f"Qwen модель не найдена: {qwen_path}")
            
            # Определяем тип квантования
            quantization = self.gpu_config.get("qwen_quantization", "4bit")
            
            self.logger.info(f"Qwen квантование: {quantization}")
            
            # Загружаем токенизатор
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                qwen_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Загружаем модель с квантованием
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    qwen_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch.float16,
                )
            elif quantization == "8bit":
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    qwen_path,
                    load_in_8bit=True,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch.float16,
                )
            else:  # float16 или float32
                dtype = torch.float16 if quantization == "float16" else torch.float32
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    qwen_path,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=dtype,
                )
            
            self.current_loaded_model = "qwen"
            self.gpu_manager.take_snapshot("after_qwen")
            
            self.logger.info("✅ Qwen загружен успешно")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки Qwen: {e}", exc_info=True)
            await self.gpu_manager.cleanup("deep")
            return False
    
    # ==================== УПРАВЛЕНИЕ ====================
    
    async def unload_current_model(self):
        """Выгрузка текущей загруженной модели"""
        if self.current_loaded_model is None:
            return
        
        self.logger.info(f"Выгрузка модели: {self.current_loaded_model}")
        
        try:
            if self.current_loaded_model == "whisper":
                del self.whisper_pipeline
                del self.whisper_processor
                self.whisper_pipeline = None
                self.whisper_processor = None
            
            elif self.current_loaded_model == "diarization":
                del self.diarization_pipeline
                self.diarization_pipeline = None
            
            elif self.current_loaded_model == "qwen":
                del self.qwen_model
                del self.qwen_tokenizer
                self.qwen_model = None
                self.qwen_tokenizer = None
            
            self.current_loaded_model = None
            await self.gpu_manager.cleanup("standard")
            
        except Exception as e:
            self.logger.warning(f"Ошибка при выгрузке модели: {e}")
    
    async def unload_all_models(self):
        """Выгрузка всех моделей"""
        self.logger.info("Выгрузка всех моделей")
        
        if self.whisper_pipeline:
            del self.whisper_pipeline
            del self.whisper_processor
        if self.diarization_pipeline:
            del self.diarization_pipeline
        if self.qwen_model:
            del self.qwen_model
            del self.qwen_tokenizer
        
        self.whisper_pipeline = None
        self.whisper_processor = None
        self.diarization_pipeline = None
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.current_loaded_model = None
        
        await self.gpu_manager.cleanup("deep")
    
    def get_loaded_models_info(self) -> Dict:
        """Информация о загруженных моделях"""
        return {
            "whisper_loaded": self.whisper_pipeline is not None,
            "diarization_loaded": self.diarization_pipeline is not None,
            "qwen_loaded": self.qwen_model is not None,
            "current_model": self.current_loaded_model,
            "device": self.device,
            "whisper_quantization": self.gpu_config.get("whisper_quantization", "unknown"),
            "qwen_quantization": self.gpu_config.get("qwen_quantization", "unknown"),
            "chunk_length_s": self.gpu_config.get("chunk_length_s"),
            "stride_length_s": self.gpu_config.get("stride_length_s"),
            "batch_size": self.whisper_batch_size,
            "max_audio_length_minutes": self.max_audio_length_minutes,
        }
