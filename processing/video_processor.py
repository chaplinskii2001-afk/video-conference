"""
Основной процессор для обработки видео и аудио файлов
Выполняет транскрипцию, диаризацию и суммаризацию
"""
import os
import re
import asyncio
import subprocess
import logging
import torch
import torchaudio
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
import yt_dlp

# Локальные модули
from config import AppConfig, GPUConfig
from processing.gpu_manager import GPUMemoryManager
from processing.model_manager import ModelManager


class VideoProcessor:
    """
    Главный класс для обработки видео/аудио файлов
    
    Этапы обработки:
    1. Подготовка аудио (извлечение из видео / конвертация)
    2. Транскрипция речи (Whisper)
    3. Диаризация спикеров (PyAnnote)
    4. Объединение транскрипции и диаризации
    5. Суммаризация текста (Qwen)
    6. Сохранение результатов
    """
    
    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Инициализация конфигурации
        self.app_config = AppConfig()
        self.gpu_config = AppConfig.get_gpu_config()
        self.model_config = AppConfig.get_model_config()
        
        # Создаем полную конфигурацию
        self.config = {
            "gpu_config": self.gpu_config,
            "model_config": self.model_config,
            "app_config": self.app_config
        }
        
        # Менеджеры
        self.gpu_manager = GPUMemoryManager(log_memory_changes=True)
        self.model_manager = ModelManager(self.config, self.gpu_manager)
        
        # Настройка torchaudio backend
        try:
            torchaudio.set_audio_backend("soundfile")
            self.logger.info("Torchaudio backend: soundfile")
        except Exception as e:
            self.logger.warning(f"Не удалось установить torchaudio backend: {e}")
        
        # Настройка CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.logger.info("=" * 60)
        self.logger.info("VideoProcessor инициализирован")
        self.logger.info(f"GPU профиль: {self.gpu_config.get('name', 'Unknown')}")
        self.logger.info(f"GPU память: {self.gpu_config.get('gpu_info', {}).get('vram_gb', 0)} GB")
        self.logger.info("=" * 60)
    
    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    
    def _get_tomsk_time(self):
        """Получение текущего времени в Томске (UTC+7)"""
        tomsk_tz = timezone(timedelta(hours=7))
        return datetime.now(tomsk_tz)
    
    def _update_progress(self, percent: int, stage: str, message: str):
        """Обновление прогресса через TaskManager"""
        if self.task_id:
            from task_manager import task_manager
            task_manager.update_progress(self.task_id, percent, stage, message)
    
    # ==================== РАБОТА С ФАЙЛАМИ ====================
    
    async def download_from_url(self, url: str, task_id: str) -> str:
        """
        Скачивание медиа файла по URL (YouTube и другие)
        """
        self.logger.info(f"Скачивание файла: {url}")
        self._update_progress(10, "download", f"Скачивание: {url}")
        
        ydl_opts = {
            "outtmpl": f"uploads/{task_id}.%(ext)s",
            "ignoreerrors": True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Находим скачанный файл
            for file in os.listdir("uploads"):
                if file.startswith(task_id) and not file.endswith(".part"):
                    file_path = f"uploads/{file}"
                    self.logger.info(f"Файл скачан: {file_path}")
                    return file_path
            
            raise Exception("Файл не найден после скачивания")
            
        except Exception as e:
            self.logger.error(f"Ошибка скачивания: {e}")
            raise
    
    def extract_audio(self, video_path: str, task_id: str) -> str:
        """
        Извлечение аудио из видео файла
        Конвертация в mono 16kHz WAV для оптимальной обработки
        """
        self.logger.info(f"Извлечение аудио из видео: {video_path}")
        audio_path = f"uploads/{task_id}.wav"
        
        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-ac", "1",  # моно
                "-ar", "16000",  # 16 kHz
                "-vn",  # без видео
                audio_path,
                "-y"  # перезапись
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            self.logger.info(f"Аудио извлечено: {audio_path}")
            return audio_path
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения аудио: {e}")
            raise
    
    def process_audio_file(self, audio_path: str, task_id: str) -> str:
        """
        Конвертация аудио файла в оптимальный формат
        """
        self.logger.info(f"Конвертация аудио: {audio_path}")
        converted_path = f"uploads/{task_id}.wav"
        
        try:
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-ac", "1",  # моно
                "-ar", "16000",  # 16 kHz
                "-vn",
                converted_path,
                "-y"
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            self.logger.info(f"Аудио конвертировано: {converted_path}")
            return converted_path
            
        except Exception as e:
            self.logger.error(f"Ошибка конвертации аудио: {e}")
            raise
    
    # ==================== ТРАНСКРИПЦИЯ ====================
    
    async def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """
        Транскрипция аудио в текст с использованием Whisper
        Возвращает список сегментов с временными метками
        """
        self.logger.info(f"Начало транскрипции: {audio_path}")
        self.gpu_manager.take_snapshot("before_transcription")
        
        # Загружаем Whisper
        self._update_progress(48, "loading_models", "Загрузка модели транскрипции (Whisper)...")
        success = await self.model_manager.load_whisper()
        if not success:
            raise Exception("Не удалось загрузить Whisper модель")
        
        self._update_progress(50, "transcription", "Транскрипция речи...")
        
        try:
            result = self.model_manager.whisper_pipeline(audio_path)
            segments = []
            
            if isinstance(result.get("chunks"), list):
                for chunk in result["chunks"]:
                    segments.append({
                        "start": chunk["timestamp"][0],
                        "end": chunk["timestamp"][1],
                        "text": chunk["text"].strip(),
                    })
            else:
                # Fallback если нет chunks
                segments.append({
                    "start": 0.0,
                    "end": None,
                    "text": result.get("text", "")
                })
            
            self.logger.info(f"Транскрипция завершена: {len(segments)} сегментов")
            self.gpu_manager.take_snapshot("after_transcription")
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Ошибка транскрипции: {e}", exc_info=True)
            raise
        finally:
            await self.model_manager.unload_current_model()
    
    # ==================== ДИАРИЗАЦИЯ ====================
    
    async def diarize_audio(self, audio_path: str) -> List[Dict]:
        """
        Диаризация аудио - определение кто и когда говорил
        Использует PyAnnote для разделения по спикерам
        """
        self.logger.info(f"Начало диаризации: {audio_path}")
        self.gpu_manager.take_snapshot("before_diarization")
        
        # Загружаем PyAnnote
        self._update_progress(58, "loading_models", "Загрузка модели диаризации (PyAnnote)...")
        success = await self.model_manager.load_diarization()
        if not success:
            raise Exception("Не удалось загрузить PyAnnote модель")
        
        self._update_progress(62, "diarization", "Определение спикеров...")
        
        try:
            # Загружаем аудио в память
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Ресемплинг если нужно
            if sample_rate != 16000:
                self.logger.info(f"Ресемплинг: {sample_rate} Hz -> 16000 Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000
                )
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Конвертация в моно если стерео
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Перенос на GPU если доступно
            if torch.cuda.is_available():
                waveform = waveform.to("cuda")
            
            self.logger.info(f"Аудио подготовлено: shape={waveform.shape}")
            
            # Запуск диаризации
            inputs = {"waveform": waveform, "sample_rate": sample_rate}
            output = self.model_manager.diarization_pipeline(inputs)
            
            # Извлечение результатов
            diarization = output.speaker_diarization
            result = []
            
            for turn, speaker in diarization:
                result.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            self.logger.info(f"Диаризация завершена: {len(result)} сегментов")
            self.gpu_manager.take_snapshot("after_diarization")
            
            # Очистка
            del waveform
            del inputs
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка диаризации: {e}", exc_info=True)
            raise
        finally:
            await self.model_manager.unload_current_model()
    
    # ==================== ОБЪЕДИНЕНИЕ ДАННЫХ ====================
    
    def align_transcription_and_diarization(
        self, 
        transcription_segments: List[Dict], 
        diarization_segments: List[Dict]
    ) -> List[Dict]:
        """
        Объединение транскрипции и диаризации
        Определяет какой спикер сказал каждый сегмент текста
        """
        self.logger.info("Объединение транскрипции и диаризации")
        aligned = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg.get("start")
            trans_end = trans_seg.get("end")
            
            if trans_start is None or trans_end is None:
                continue
            
            # Находим пересечения со спикерами
            speaker_overlaps = {}
            
            for diar_seg in diarization_segments:
                overlap_start = max(trans_start, diar_seg["start"])
                overlap_end = min(trans_end, diar_seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > 0:
                    speaker = diar_seg["speaker"]
                    speaker_overlaps[speaker] = speaker_overlaps.get(speaker, 0) + overlap_duration
            
            # Определяем доминирующего спикера
            if speaker_overlaps:
                dominant_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
            else:
                dominant_speaker = "UNKNOWN"
            
            aligned.append({
                "start": trans_start,
                "end": trans_end,
                "speaker": dominant_speaker,
                "text": trans_seg["text"]
            })
        
        speakers_count = len(set(seg["speaker"] for seg in aligned))
        self.logger.info(f"Объединение завершено: {len(aligned)} сегментов, {speakers_count} спикеров")
        
        return aligned
    
    # ==================== СУММАРИЗАЦИЯ ====================
    
    def split_text_into_chunks(self, text: str, max_chars: int = 14000) -> List[str]:
        """
        Разбивает длинный текст на части для обработки
        Старается разбивать по предложениям
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # Если предложение слишком длинное - разбиваем принудительно
            if sentence_len > max_chars:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                for i in range(0, sentence_len, max_chars):
                    chunks.append(sentence[i:i + max_chars])
                continue
            
            # Если добавление превысит лимит - сохраняем текущий чанк
            if current_length + sentence_len > max_chars:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    async def summarize_chunk(
        self, 
        text: str, 
        summary_type: str = "standard",
        max_tokens: int = 800
    ) -> str:
        """
        Суммаризация одного чанка текста
        
        Исправления обрезания суммаризации:
        - max_length при токенизации увеличен с 12000 до 24000
          Позволяет модели видеть больше контекста для анализа
        - Добавлен min_new_tokens для гарантии минимального выхода
          Гарантирует что модель сгенерирует не менее 80% от max_tokens
        - Логирование длины input и output для отладки
        """
        # Шаблоны для разных типов суммаризации
        if summary_type == "standard":
            system_message = """Ты - секретарь, оформляющий протокол видео на русском языке. 
Используй следующий шаблон:

Общая тематика видео:
Основные темы:
Ключевые тезисы:
Итоги:

Требования:
1. Строго придерживайся структуры шаблона
2. Выводи только пункты из шаблона
3. Абсолютная грамотность
4. Используй только русский язык"""
        else:  # protocol
            system_message = """Ты - секретарь, оформляющий протокол конференции. 
Шаблон:

Дата проведения:
Присутствуют:
Повестка дня:
Рассмотрены вопросы и решения:
Итоги:

Требования:
1. Строго придерживайся шаблона
2. Только русский язык"""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Проанализируй текст и создай содержание:\n{text}"}
        ]
        
        prompt = self.model_manager.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.model_manager.qwen_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=24000
        ).to(self.model_manager.qwen_model.device)
        
        input_length = len(inputs.input_ids[0])
        self.logger.info(f"Суммаризация чанка: input_tokens={input_length}, max_new_tokens={max_tokens}")
        
        with torch.no_grad():
            outputs = self.model_manager.qwen_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=max(100, int(max_tokens * 0.8)),
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
            )
        
        generated_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        self.logger.info(f"Сгенерировано токенов: {len(generated_ids)} (ожидалось {max_tokens})")
        
        summary = self.model_manager.qwen_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()
        
        return summary
    
    async def merge_summaries(
        self,
        summaries: List[str],
        summary_type: str = "standard"
    ) -> str:
        """
        Объединение нескольких суммаризаций в одну
        
        Исправления обрезания суммаризации:
        - max_length при токенизации увеличен с 24000 до 32000
          Позволяет модели видеть больше контекста при объединении
        - Добавлен min_new_tokens для гарантии минимального выхода
          Гарантирует что модель сгенерирует не менее 80% от max_new_tokens (5600+)
        - Логирование длины input и output для отладки
        """
        if summary_type == "standard":
            system_message = "Объедини суммаризации в ЕДИНЫЙ структурированный документ. Убери дубликаты."
        else:
            system_message = "Объедини протоколы в ЕДИНЫЙ протокол конференции. Объедини списки присутствующих и вопросы."
        
        combined_text = "\n\n".join(summaries)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Объедини текст:\n{combined_text}"}
        ]
        
        prompt = self.model_manager.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        max_new_tokens = 7000
        inputs = self.model_manager.qwen_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=32000
        ).to(self.model_manager.qwen_model.device)
        
        input_length = len(inputs.input_ids[0])
        self.logger.info(f"Объединение суммаризаций: input_tokens={input_length}, max_new_tokens={max_new_tokens}")
        
        with torch.no_grad():
            outputs = self.model_manager.qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max(500, int(max_new_tokens * 0.8)),
                do_sample=False,
                repetition_penalty=1.05,
            )
        
        generated_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        self.logger.info(f"Сгенерировано токенов: {len(generated_ids)} (ожидалось {max_new_tokens})")
        
        final_summary = self.model_manager.qwen_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()
        
        return final_summary
    
    async def summarize_text(
        self,
        text: str,
        summary_type: str = "standard"
    ) -> str:
        """
        Главный метод суммаризации
        Разбивает текст на части, суммаризует каждую и объединяет
        """
        if not text.strip():
            raise ValueError("Текст пуст")
        
        self.logger.info(f"Начало суммаризации ({summary_type}): {len(text)} символов")
        self.gpu_manager.take_snapshot("before_summarization")
        
        # Загружаем Qwen
        self._update_progress(80, "loading_models", "Загрузка модели суммаризации (Qwen)...")
        success = await self.model_manager.load_qwen()
        if not success:
            raise Exception("Не удалось загрузить Qwen модель")
        
        self._update_progress(85, "summarization", "Создание документа...")
        
        try:
            # Разбиваем текст на части
            max_chars = AppConfig.SUMMARIZATION_MAX_CHARS
            chunks = self.split_text_into_chunks(text, max_chars=max_chars)
            total_chunks = len(chunks)
            
            self.logger.info(f"Текст разбит на {total_chunks} частей")
            
            # Суммаризуем каждую часть
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                self._update_progress(
                    85 + int(10 / total_chunks * i),
                    "summarization",
                    f"Суммаризация части {i+1}/{total_chunks}"
                )
                
                max_tokens = AppConfig.SUMMARY_MAX_NEW_TOKENS.get(summary_type, 800)
                summary = await self.summarize_chunk(chunk, summary_type, max_tokens)
                chunk_summaries.append(summary)
                
                # Легкая очистка между чанками
                await self.gpu_manager.cleanup("light")
            
            # Если был только один чанк - возвращаем его
            if len(chunk_summaries) == 1:
                final_summary = chunk_summaries[0]
            else:
                # Объединяем суммаризации
                self.logger.info("Объединение суммаризаций")
                final_summary = await self.merge_summaries(chunk_summaries, summary_type)
            
            self.logger.info(f"Суммаризация завершена: {len(final_summary)} символов")
            self.gpu_manager.take_snapshot("after_summarization")
            
            return final_summary
            
        except Exception as e:
            self.logger.error(f"Ошибка суммаризации: {e}", exc_info=True)
            raise
        finally:
            await self.model_manager.unload_current_model()
    
    # ==================== ФОРМАТИРОВАНИЕ И СОХРАНЕНИЕ ====================
    
    def format_transcription(self, segments: List[Dict]) -> str:
        """
        Форматирование транскрипции для удобного чтения
        """
        formatted = "# ТРАНСКРИПЦИЯ\n\n"
        current_speaker = None
        
        for segment in segments:
            speaker = segment["speaker"]
            text = segment["text"]
            
            if speaker != current_speaker:
                formatted += f"\n**{speaker}**:\n"
                current_speaker = speaker
            
            formatted += f"{text} "
        
        return formatted
    
    def add_metadata(
        self,
        content: str,
        task_id: str,
        processing_time: float,
        media_type: str,
        stats: Dict
    ) -> str:
        """
        Добавление метаданных к результату
        """
        tomsk_time = self._get_tomsk_time()
        
        metadata = f"""# Отчет обработки

**ID задачи**: {task_id}
**Дата**: {tomsk_time.strftime('%Y-%m-%d %H:%M:%S')} (Томск, UTC+7)
**Время обработки**: {processing_time:.2f} мин
**Тип медиа**: {media_type}
**Сегментов**: {stats.get('segments_count', 0)}
**Спикеров**: {stats.get('speakers_count', 0)}

**GPU Статистика**:
- Пиковое использование: {stats.get('peak_gpu_usage', 0):.2f} GB ({stats.get('peak_gpu_percent', 0):.1f}%)
- Пиковый этап: {stats.get('peak_stage', 'unknown')}

---

"""
        return metadata + content
    
    def save_result(self, file_path: str, content: str):
        """
        Сохранение результата в файл
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Результат сохранен: {file_path}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения файла {file_path}: {e}")
            raise
    
    # ==================== ГЛАВНЫЙ ПАЙПЛАЙН ====================
    
    async def process_media(
        self,
        file_path: str,
        task_id: str,
        media_type: str = "video",
        summary_type: str = "standard"
    ) -> Dict:
        """
        Главный метод обработки медиа файла
        
        Выполняет полный цикл:
        1. Подготовка аудио
        2. Транскрипция
        3. Диаризация
        4. Объединение
        5. Суммаризация
        6. Сохранение
        """
        self.logger.info("=" * 60)
        self.logger.info(f"НАЧАЛО ОБРАБОТКИ ЗАДАЧИ: {task_id}")
        self.logger.info(f"Тип: {media_type}, Суммаризация: {summary_type}")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        audio_path = None
        
        try:
            # Очистка перед началом
            await self.gpu_manager.cleanup("standard")
            self.gpu_manager.take_snapshot("initial")
            
            # 1. ПОДГОТОВКА АУДИО
            self._update_progress(20, "audio_extraction", "Извлечение аудиодорожки...")
            if media_type == "audio":
                audio_path = self.process_audio_file(file_path, task_id)
            else:
                audio_path = self.extract_audio(file_path, task_id)
            
            # 2. ЗАГРУЗКА МОДЕЛЕЙ И ТРАНСКРИПЦИЯ
            self._update_progress(35, "loading_models", "Загрузка моделей ИИ...")
            self._update_progress(45, "transcription", "Транскрипция речи...")
            transcription_segments = await self.transcribe_audio(audio_path)
            
            # 3. ДИАРИЗАЦИЯ
            self._update_progress(60, "diarization", "Определение спикеров...")
            diarization_segments = await self.diarize_audio(audio_path)
            
            # 4. ОБЪЕДИНЕНИЕ
            self._update_progress(75, "merging", "Объединение результатов...")
            aligned_segments = self.align_transcription_and_diarization(
                transcription_segments,
                diarization_segments
            )
            
            # Извлекаем полный текст и статистику
            full_text = " ".join([seg["text"] for seg in aligned_segments])
            speakers_count = len(set(seg["speaker"] for seg in aligned_segments))
            
            # 5. СУММАРИЗАЦИЯ
            self._update_progress(85, "summarization", "Создание документа...")
            summary = await self.summarize_text(full_text, summary_type)
            
            # 6. ФОРМАТИРОВАНИЕ И СОХРАНЕНИЕ
            self._update_progress(95, "saving", "Сохранение результатов...")
            
            formatted_transcription = self.format_transcription(aligned_segments)
            processing_time = (datetime.now() - start_time).total_seconds() / 60
            
            # Получаем статистику GPU
            gpu_stats = self.gpu_manager.get_memory_stats()
            
            stats = {
                "segments_count": len(aligned_segments),
                "speakers_count": speakers_count,
                "peak_gpu_usage": gpu_stats.get("peak_usage_gb", 0),
                "peak_gpu_percent": gpu_stats.get("peak_usage_percent", 0),
                "peak_stage": gpu_stats.get("peak_stage", "unknown")
            }
            
            # Добавляем метаданные
            formatted_transcription = self.add_metadata(
                formatted_transcription,
                task_id,
                processing_time,
                media_type,
                stats
            )
            
            # Сохраняем файлы
            self.save_result(
                f"results/{task_id}_transcription.md",
                formatted_transcription
            )
            self.save_result(
                f"results/{task_id}_summary.md",
                summary
            )
            
            # Выводим сводку по памяти
            self.gpu_manager.log_memory_summary()
            
            self._update_progress(100, "completed", "Обработка завершена")
            
            self.logger.info("=" * 60)
            self.logger.info(f"ЗАДАЧА ЗАВЕРШЕНА: {task_id}")
            self.logger.info(f"Время обработки: {processing_time:.2f} мин")
            self.logger.info(f"Сегментов: {len(aligned_segments)}, Спикеров: {speakers_count}")
            self.logger.info("=" * 60)
            
            return {
                "task_id": task_id,
                "summary": summary,
                "transcription_length": len(formatted_transcription),
                "segments_count": len(aligned_segments),
                "speakers_count": speakers_count,
                "processing_time_minutes": round(processing_time, 2),
                "media_type": media_type,
                "gpu_peak_usage_gb": gpu_stats.get("peak_usage_gb", 0),
            }
            
        except Exception as e:
            self.logger.error("=" * 60)
            self.logger.error(f"КРИТИЧЕСКАЯ ОШИБКА В ЗАДАЧЕ: {task_id}")
            self.logger.error(f"Ошибка: {e}", exc_info=True)
            self.logger.error("=" * 60)
            self._update_progress(0, "error", str(e))
            raise
            
        finally:
            # Очистка
            await self.model_manager.unload_all_models()
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    self.logger.info(f"Временный файл удален: {audio_path}")
                except Exception as e:
                    self.logger.warning(f"Не удалось удалить {audio_path}: {e}")
            
            await self.gpu_manager.cleanup("deep")
            self.gpu_manager.clear_snapshots()
    
    # ==================== СЛУЖЕБНЫЕ МЕТОДЫ ====================
    
    def check_gpu(self) -> bool:
        """Проверка доступности GPU"""
        return torch.cuda.is_available()
    
    def get_gpu_memory_info(self) -> Dict:
        """Получить информацию о памяти GPU"""
        return self.gpu_manager.get_memory_info()
    
    def get_system_info(self) -> Dict:
        """Получить информацию о системе"""
        return {
            "gpu_available": torch.cuda.is_available(),
            "gpu_profile": self.gpu_config.get("name", "Unknown"),
            "gpu_vram_gb": self.gpu_config.get("gpu_info", {}).get("vram_gb", 0),
            "device": self.model_manager.device if hasattr(self, "model_manager") else "unknown"
        }
    
    async def force_gpu_cleanup(self):
        """Принудительная очистка GPU (для совместимости)"""
        await self.gpu_manager.cleanup("deep")
    
    async def unload_current_model(self):
        """Выгрузка текущей модели (для совместимости)"""
        if hasattr(self, "model_manager"):
            await self.model_manager.unload_current_model()
    
    @property
    def current_loaded_model(self):
        """Текущая загруженная модель (для совместимости)"""
        if hasattr(self, "model_manager"):
            return self.model_manager.current_loaded_model
        return None
