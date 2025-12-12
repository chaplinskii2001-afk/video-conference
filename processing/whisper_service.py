"""
Сервис для работы с моделью Whisper.
Отвечает только за транскрипцию аудио в текст.
"""
import os
import logging
from typing import List, Dict, Optional
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)
import gc
from config import get_settings, get_model_config


logger = logging.getLogger(__name__)


class WhisperService:
    """Сервис транскрипции аудио с помощью Whisper"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.model = None
        self.processor = None
        self.model_id = get_settings().WHISPER_MODEL_ID
        self.model_config = get_model_config()
    
    async def initialize(self) -> None:
        """Инициализация модели Whisper"""
        if self.model is not None:
            return
            
        logger.info(f"Инициализация Whisper модели: {self.model_id}")
        
        await self._cleanup_memory()
        
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # Загружаем процессор и модель
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
            )
            
            # Создаем пайплайн
            self.model = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=self.model_config.WHISPER_CHUNK_LENGTH,
                stride_length_s=self.model_config.WHISPER_STRIDE_LENGTH,
                return_timestamps=True,
            )
            
            logger.info("Whisper модель загружена успешно")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки Whisper модели: {e}")
            await self._cleanup_memory()
            raise
    
    async def transcribe(self, audio_path: str) -> List[Dict]:
        """
        Транскрипция аудио в текст
        
        Args:
            audio_path: Путь к аудио файлу
            
        Returns:
            Список сегментов с временными метками и текстом
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
        
        await self.initialize()
        
        try:
            logger.info(f"Начало транскрипции: {audio_path}")
            
            # Выполняем транскрипцию
            result = self.model(audio_path)
            
            segments = []
            if isinstance(result.get("chunks"), list):
                for chunk in result["chunks"]:
                    if isinstance(chunk.get("timestamp"), (list, tuple)) and len(chunk["timestamp"]) >= 2:
                        segments.append({
                            "start": float(chunk["timestamp"][0]),
                            "end": float(chunk["timestamp"][1]),
                            "text": chunk["text"].strip(),
                        })
                    else:
                        # Fallback для случаев без временных меток
                        segments.append({
                            "start": 0.0,
                            "end": None,
                            "text": chunk.get("text", "").strip(),
                        })
            else:
                # Обработка случая с полным текстом без сегментов
                segments.append({
                    "start": 0.0,
                    "end": None,
                    "text": result.get("text", "").strip(),
                })
            
            logger.info(f"Транскрипция завершена. Сегментов: {len(segments)}")
            return segments
            
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {e}")
            raise
    
    async def unload(self) -> None:
        """Выгрузка модели из памяти"""
        if self.model:
            logger.info("Выгрузка Whisper модели из памяти")
            del self.model
            self.model = None
        
        if self.processor:
            del self.processor
            self.processor = None
            
        await self._cleanup_memory()
    
    async def _cleanup_memory(self) -> None:
        """Очистка памяти GPU"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Ошибка очистки памяти: {e}")
    
    def is_loaded(self) -> bool:
        """Проверка, загружена ли модель"""
        return self.model is not None
    
    def get_model_info(self) -> Dict:
        """Информация о модели"""
        return {
            "model_id": self.model_id,
            "loaded": self.is_loaded(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "quantization": "8-bit",
        }