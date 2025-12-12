"""
Обновленный VideoProcessor с разделением ответственности.
Использует отдельные сервисы для каждой AI модели.
"""
import os
import logging
import subprocess
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

import torch
import torchaudio
import pynvml
import psutil
import gc

from processing.whisper_service import WhisperService
from processing.diarization_service import DiarizationService
from processing.summarization_service import SummarizationService
from config import get_settings, get_security_config


logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Основной процессор видео с использованием отдельных сервисов.
    Координирует работу между сервисами и управляет общим состоянием.
    """
    
    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id
        self.settings = get_settings()
        self.security_config = get_security_config()
        
        # Инициализируем сервисы
        self.whisper_service = WhisperService()
        self.diarization_service = DiarizationService()
        self.summarization_service = SummarizationService()
        
        # Настройка PyTorch
        self._setup_torch_config()
        
        logger.info("VideoProcessor инициализирован с новой архитектурой")
    
    def _setup_torch_config(self) -> None:
        """Настройка стабильности PyTorch/CUDA"""
        if torch.cuda.is_available():
            # Отключаем бенчмарк для избежания ошибок памяти
            torch.backends.cudnn.benchmark = False
            # Разрешаем TF32 для производительности
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Настройка torchaudio backend
        try:
            torchaudio.set_audio_backend("soundfile")
            logger.info("Torchaudio backend: soundfile")
        except Exception as e:
            logger.warning(f"Не удалось настроить torchaudio backend: {e}")
    
    def _get_tomsk_time(self) -> datetime:
        """Получение текущего времени в Томске (UTC+7)"""
        tomsk_tz = timezone(timedelta(hours=self.settings.TOMSK_TIMEZONE_OFFSET))
        return datetime.now(tomsk_tz)
    
    def _update_progress(self, percent: int, stage: str, message: str) -> None:
        """Обновление прогресса через TaskManager"""
        if self.task_id:
            from task_manager import task_manager
            task_manager.update_progress(self.task_id, percent, stage, message)
    
    # ================ УПРАВЛЕНИЕ ПАМЯТЬЮ ================
    
    def get_gpu_memory_info(self) -> Dict:
        """Информация о памяти GPU"""
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "total_gb": round(info.total / 1024**3, 2),
                "used_gb": round(info.used / 1024**3, 2),
                "free_gb": round(info.free / 1024**3, 2),
                "usage_percent": round((info.used / info.total) * 100, 1),
            }
        except Exception as e:
            logger.warning(f"Не удалось получить информацию о GPU: {e}")
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "usage_percent": 0}
    
    def get_system_memory_info(self) -> Dict:
        """Информация о системной памяти"""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": round(memory.total / 1024**3, 2),
                "available_gb": round(memory.available / 1024**3, 2),
                "used_gb": round(memory.used / 1024**3, 2),
                "usage_percent": round(memory.percent, 1),
            }
        except Exception as e:
            logger.warning(f"Не удалось получить информацию о системной памяти: {e}")
            return {"total_gb": 0, "available_gb": 0, "used_gb": 0, "usage_percent": 0}
    
    async def force_gpu_cleanup(self) -> None:
        """Принудительная очистка памяти GPU"""
        logger.info("Принудительная очистка памяти GPU...")
        try:
            # Выгружаем все модели
            await self.whisper_service.unload()
            await self.diarization_service.unload()
            await self.summarization_service.unload()
            
            # Очищаем память
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            await asyncio.sleep(1)
            logger.info("Очистка памяти завершена")
        except Exception as e:
            logger.warning(f"Ошибка при очистке GPU памяти: {e}")
    
    # ================ ОБРАБОТКА МЕДИА ================
    
    async def download_from_url(self, url: str, task_id: str) -> str:
        """Скачивание медиа по URL"""
        logger.info(f"Скачивание по URL: {url}")
        self._update_progress(10, "download", f"Скачивание: {url}")
        
        import yt_dlp
        ydl_opts = {
            "outtmpl": f"{self.settings.UPLOAD_DIR}/{task_id}.%(ext)s",
            "ignoreerrors": True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Ищем скачанный файл
            for file in os.listdir(self.settings.UPLOAD_DIR):
                if file.startswith(task_id) and not file.endswith(".part"):
                    return f"{self.settings.UPLOAD_DIR}/{file}"
            
            raise Exception("Файл не найден после скачивания")
        except Exception as e:
            logger.error(f"Ошибка скачивания: {e}")
            raise
    
    def extract_audio(self, video_path: str, task_id: str) -> str:
        """Извлечение аудио из видео"""
        logger.info(f"Извлечение аудио: {video_path}")
        audio_path = f"{self.settings.UPLOAD_DIR}/{task_id}.wav"
        
        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-ac", str(self.settings.AUDIO_CHANNELS),  # моно
                "-ar", str(self.settings.AUDIO_SAMPLE_RATE),  # 16kHz
                "-vn",  # без видео
                audio_path,
                "-y",
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return audio_path
        except Exception as e:
            logger.error(f"Ошибка ffmpeg: {e}")
            raise
    
    def process_audio_file(self, audio_path: str, task_id: str) -> str:
        """Конвертация аудио файла в нужный формат"""
        logger.info(f"Конвертация аудио: {audio_path}")
        converted_path = f"{self.settings.UPLOAD_DIR}/{task_id}.wav"
        
        try:
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-ac", str(self.settings.AUDIO_CHANNELS),
                "-ar", str(self.settings.AUDIO_SAMPLE_RATE),
                "-vn",
                converted_path,
                "-y",
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return converted_path
        except Exception as e:
            logger.error(f"Ошибка конвертации аудио: {e}")
            raise
    
    # ================ АЛГОРИТМЫ ОБРАБОТКИ ================
    
    def align_asr_and_diarization(
        self, 
        asr_segments: List[Dict], 
        diar_segments: List[Dict]
    ) -> List[Dict]:
        """
        Объединение результатов транскрипции и диаризации по времени
        
        Args:
            asr_segments: Сегменты от Whisper
            diar_segments: Сегменты от PyAnnote
            
        Returns:
            Объединенные сегменты с указанием спикеров
        """
        aligned = []
        
        for asr in asr_segments:
            asr_start = asr.get("start")
            asr_end = asr.get("end")
            
            if asr_start is None or asr_end is None:
                continue
            
            # Ищем пересекающиеся сегменты диаризации
            overlap_speakers = {}
            for dia in diar_segments:
                overlap_start = max(asr_start, dia["start"])
                overlap_end = min(asr_end, dia["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > 0:
                    overlap_speakers[dia["speaker"]] = (
                        overlap_speakers.get(dia["speaker"], 0) + overlap_duration
                    )
            
            # Определяем доминирующего спикера
            if overlap_speakers:
                dominant_speaker = max(overlap_speakers, key=overlap_speakers.get)
            else:
                dominant_speaker = "UNKNOWN"
            
            aligned.append({
                "start": asr_start,
                "end": asr_end,
                "speaker": dominant_speaker,
                "text": asr["text"],
            })
        
        return aligned
    
    def format_transcription(self, merged_data: List[Dict]) -> str:
        """Форматирование транскрипции для вывода"""
        formatted = "ТРАНСКРИПЦИЯ\n" + "=" * 20 + "\n\n"
        current_speaker = None
        
        for segment in merged_data:
            speaker = segment["speaker"]
            text = segment["text"]
            
            if speaker != current_speaker:
                formatted += f"\n**{speaker}**:\n"
                current_speaker = speaker
            
            formatted += f"{text} "
        
        return formatted
    
    def add_processing_metadata(
        self, 
        content: str, 
        task_id: str, 
        processing_time: float, 
        media_type: str
    ) -> str:
        """Добавление метаданных обработки"""
        tomsk_time = self._get_tomsk_time()
        meta = f"""# Отчет обработки
ID: {task_id}
Дата: {tomsk_time.strftime('%Y-%m-%d %H:%M:%S')}
Время обработки: {processing_time:.2f} мин
Тип медиа: {media_type}
--------------------------------------------
"""
        return meta + content
    
    def save_formatted_text(
        self, 
        file_path: str, 
        content: str, 
        file_type: str, 
        is_markdown: bool = False
    ) -> None:
        """Сохранение текста в файл"""
        if is_markdown and not file_path.lower().endswith(".md"):
            file_path = file_path.rsplit(".", 1)[0] + ".md"
        
        try:
            with open(file_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            logger.info(f"Файл сохранен: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения {file_path}: {e}")
            raise
    
    # ================ ОСНОВНОЙ ПАЙПЛАЙН ================
    
    async def process_media(
        self,
        file_path: str,
        task_id: str,
        media_type: str = "video",
        summary_type: str = "standard",
    ) -> Dict:
        """
        Основной пайплайн обработки медиа
        
        Args:
            file_path: Путь к файлу
            task_id: ID задачи
            media_type: Тип медиа (video/audio)
            summary_type: Тип суммаризации
            
        Returns:
            Результаты обработки
        """
        logger.info(f"Начало обработки задачи {task_id}")
        start_time = datetime.now()
        audio_path = None
        
        try:
            await self.force_gpu_cleanup()
            
            # 1. Подготовка аудио
            self._update_progress(20, "audio_extraction", "Подготовка аудио...")
            if media_type == "audio":
                audio_path = self.process_audio_file(file_path, task_id)
            else:
                audio_path = self.extract_audio(file_path, task_id)
            
            # 2. Транскрипция (Whisper)
            self._update_progress(30, "transcription", "Транскрипция речи...")
            transcription = await self.whisper_service.transcribe(audio_path)
            
            # 3. Диаризация (PyAnnote)
            self._update_progress(50, "diarization", "Определение спикеров...")
            diarization = await self.diarization_service.diarize(audio_path)
            
            # 4. Объединение результатов
            self._update_progress(65, "merging", "Сведение результатов...")
            merged = self.align_asr_and_diarization(transcription, diarization)
            
            # Собираем полный текст
            full_text = " ".join([seg["text"] for seg in merged])
            speakers_count = len(set([seg["speaker"] for seg in merged if seg["speaker"] != "UNKNOWN"]))
            
            # 5. Суммаризация (Qwen)
            self._update_progress(75, "summarization", "Создание краткого содержания...")
            if len(full_text) > self.settings.MAX_CHUNK_SIZE:
                summary = await self.summarization_service.smart_summarize_long_text(
                    full_text, summary_type
                )
            else:
                summary = await self.summarization_service.summarize(
                    full_text, summary_type
                )
            
            # 6. Форматирование и сохранение
            self._update_progress(90, "saving", "Сохранение результатов...")
            formatted_transcription = self.format_transcription(merged)
            processing_time = (datetime.now() - start_time).total_seconds() / 60
            
            # Добавляем метаданные
            formatted_transcription = self.add_processing_metadata(
                formatted_transcription, task_id, processing_time, media_type
            )
            
            # Сохраняем файлы
            transcription_path = f"{self.settings.RESULTS_DIR}/{task_id}_transcription.md"
            summary_path = f"{self.settings.RESULTS_DIR}/{task_id}_summary.md"
            
            self.save_formatted_text(
                transcription_path, formatted_transcription, "Транскрипция", is_markdown=True
            )
            self.save_formatted_text(
                summary_path, summary, "Краткое содержание", is_markdown=True
            )
            
            self._update_progress(100, "completed", "Готово")
            
            return {
                "task_id": task_id,
                "summary": summary,
                "transcription_length": len(formatted_transcription),
                "segments_count": len(merged),
                "speakers_count": speakers_count,
                "processing_time_minutes": round(processing_time, 2),
                "media_type": media_type,
                "summary_type": summary_type,
            }
            
        except Exception as e:
            logger.error(f"ОШИБКА ОБРАБОТКИ: {e}", exc_info=True)
            self._update_progress(0, "error", str(e))
            raise
        finally:
            # Очистка
            await self.whisper_service.unload()
            await self.diarization_service.unload()
            await self.summarization_service.unload()
            
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл: {e}")
            
            await self.force_gpu_cleanup()
    
    # ================ СТАТУС И ИНФОРМАЦИЯ ================
    
    def check_gpu(self) -> bool:
        """Проверка доступности GPU"""
        return torch.cuda.is_available()
    
    def get_all_models_status(self) -> Dict:
        """Статус всех моделей"""
        return {
            "whisper": self.whisper_service.get_model_info(),
            "diarization": self.diarization_service.get_model_info(),
            "summarization": self.summarization_service.get_model_info(),
        }
    
    async def health_check(self) -> Dict:
        """Проверка здоровья системы"""
        gpu_status = self.check_gpu()
        memory_info = self.get_gpu_memory_info()
        system_info = self.get_system_memory_info()
        
        return {
            "gpu_available": gpu_status,
            "gpu_memory": memory_info,
            "system_memory": system_info,
            "models_loaded": {
                "whisper": self.whisper_service.is_loaded(),
                "diarization": self.diarization_service.is_loaded(),
                "summarization": self.summarization_service.is_loaded(),
            }
        }