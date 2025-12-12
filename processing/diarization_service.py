"""
Сервис для работы с PyAnnote диаризацией.
Отвечает за определение спикеров в аудио.
"""
import os
import logging
from typing import List, Dict, Optional, Union
import torch
import torchaudio
from pyannote.audio import Pipeline
import gc
from config import get_settings, get_model_config


logger = logging.getLogger(__name__)


class DiarizationService:
    """Сервис диаризации спикеров с помощью PyAnnote"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.pipeline = None
        self.model_id = get_settings().DIARIZATION_MODEL_ID
        self.model_config = get_model_config()
        
        # Настройка torchaudio backend
        self._setup_torchaudio_backend()
    
    def _setup_torchaudio_backend(self) -> None:
        """Настройка безопасного backend для torchaudio"""
        try:
            torchaudio.set_audio_backend("soundfile")
            logger.info("Torchaudio backend установлен: soundfile")
        except Exception as e:
            logger.warning(f"Не удалось установить torchaudio backend: {e}")
    
    async def initialize(self) -> None:
        """Инициализация модели диаризации"""
        if self.pipeline is not None:
            return
            
        logger.info(f"Инициализация PyAnnote модели: {self.model_id}")
        
        await self._cleanup_memory()
        
        try:
            # Загружаем пайплайн PyAnnote
            self.pipeline = Pipeline.from_pretrained(
                self.model_id,
                token=self.hf_token,
                cache_dir=get_settings().PYANNOTE_CACHE_PATH,
            )
            
            # Перенос на GPU если доступен
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                logger.info("PyAnnote перемещен на GPU")
            
            logger.info("PyAnnote модель загружена успешно")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки PyAnnote модели: {e}")
            logger.error(
                "Убедитесь, что HF_TOKEN верен и вы приняли условия использования модели на HuggingFace"
            )
            await self._cleanup_memory()
            raise
    
    async def diarize(self, audio_path: str) -> List[Dict]:
        """
        Определение спикеров в аудио
        
        Args:
            audio_path: Путь к аудио файлу
            
        Returns:
            Список сегментов с информацией о спикерах
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
        
        await self.initialize()
        
        try:
            logger.info(f"Начало диаризации: {audio_path}")
            
            # Загружаем аудио в память
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Проверяем и приводим к нужному формату
            waveform, sample_rate = self._prepare_audio(waveform, sample_rate)
            
            # Перемещаем на GPU если нужно
            if torch.cuda.is_available():
                waveform = waveform.to("cuda")
            
            # Выполняем диаризацию
            inputs = {"waveform": waveform, "sample_rate": sample_rate}
            output = self.pipeline(inputs)
            
            # Извлекаем результат
            diarization = output.speaker_diarization
            
            # Преобразуем в удобный формат
            segments = []
            for turn, speaker in diarization:
                segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": speaker,
                })
            
            # Очищаем память
            del waveform
            del inputs
            
            logger.info(f"Диаризация завершена. Сегментов: {len(segments)}")
            return segments
            
        except Exception as e:
            logger.error(f"Ошибка диаризации: {e}")
            raise
    
    def _prepare_audio(self, waveform: torch.Tensor, sample_rate: int) -> tuple:
        """
        Подготовка аудио для диаризации
        
        Args:
            waveform: Тензор аудио
            sample_rate: Частота дискретизации
            
        Returns:
            Подготовленный тензор и частота
        """
        # Если частота не 16kHz, делаем resample
        target_rate = get_settings().AUDIO_SAMPLE_RATE
        if sample_rate != target_rate:
            logger.info(f"Resampling аудио: {sample_rate} Hz → {target_rate} Hz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_rate
            )
            waveform = resampler(waveform)
            sample_rate = target_rate
        
        # Если стерео, конвертируем в моно (усреднение каналов)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform, sample_rate
    
    async def unload(self) -> None:
        """Выгрузка модели из памяти"""
        if self.pipeline:
            logger.info("Выгрузка PyAnnote модели из памяти")
            del self.pipeline
            self.pipeline = None
            
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
        return self.pipeline is not None
    
    def get_model_info(self) -> Dict:
        """Информация о модели"""
        return {
            "model_id": self.model_id,
            "loaded": self.is_loaded(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "sample_rate": get_settings().AUDIO_SAMPLE_RATE,
        }