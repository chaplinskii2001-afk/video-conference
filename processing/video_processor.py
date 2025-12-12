import os
import asyncio
import subprocess
import logging
import re
import torch
import torchaudio
from typing import Dict, List
from datetime import datetime, timezone, timedelta
from transformers import (
    pipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import yt_dlp
import aiohttp
import gc
import psutil
import pynvml
import numpy as np

# Импорт PyAnnote
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, task_id: str = None):
        self.task_id = task_id
        self.hf_token = os.getenv("HF_TOKEN")

        # Переменные для моделей
        self.whisper_model = None
        self.diarization_pipeline = None
        self.qwen_model = None
        self.qwen_tokenizer = None

        self.current_loaded_model = None

        # Пути
        self.whisper_model_path = "/app/models/whisper"
        self.pyannote_cache_path = "/app/models/pyannote"
        self.qwen_model_path = "/app/models/qwen"

        # ID моделей
        self.whisper_model_id = "bond005/whisper-podlodka-turbo"
        # Используем Community-1
        self.diarization_model_id = "pyannote/speaker-diarization-3.1"

        # Настройка стабильности PyTorch/CUDA
        if torch.cuda.is_available():
            # Отключаем бенчмарк cudnn для избежания редких ошибок памяти/segfault
            torch.backends.cudnn.benchmark = False
            # Разрешаем TF32 для производительности (хотя pyannote может ругаться, это безопасно)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Принудительно устанавливаем backend для аудио (избегаем torchcodec/ffmpeg binding segfaults)
        try:
            torchaudio.set_audio_backend("soundfile")
            logger.info("Torchaudio backend set to: soundfile")
        except Exception as e:
            logger.warning(f"Could not set torchaudio backend: {e}")

        logger.info(
            f"Инициализация VideoProcessor. Diarization: {self.diarization_model_id}"
        )

    def _get_tomsk_time(self):
        """Получение текущего времени в Томске (UTC+7)"""
        tomsk_tz = timezone(timedelta(hours=7))
        return datetime.now(tomsk_tz)

    def _update_progress(self, percent: int, stage: str, message: str):
        """Обновляет прогресс через TaskManager"""
        if self.task_id:
            from task_manager import task_manager

            task_manager.update_progress(self.task_id, percent, stage, message)

    def get_gpu_memory_info(self):
        """Получить информацию о памяти GPU"""
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "total_gb": info.total / 1024**3,
                "used_gb": info.used / 1024**3,
                "free_gb": info.free / 1024**3,
                "usage_percent": (info.used / info.total) * 100,
            }
        except Exception as e:
            logger.warning(f"Не удалось получить информацию о GPU: {e}")
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "usage_percent": 0}

    async def force_gpu_cleanup(self):
        """Очистка памяти GPU"""
        logger.info("Очистка памяти GPU...")
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"Ошибка при очистке GPU памяти: {e}")

    async def light_memory_cleanup(self):
        """Легкая очистка памяти без выгрузки модели"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        await asyncio.sleep(0.1)

    # ================= ЗАГРУЗКА МОДЕЛЕЙ =================

    async def load_whisper_model(self):
        """Загрузка модели Whisper с 8-битным квантованием"""
        if self.whisper_model is not None and self.current_loaded_model == "whisper":
            return

        await self.force_gpu_cleanup()
        logger.info(f"Загрузка модели Whisper '{self.whisper_model_id}'...")
        self._update_progress(0, "loading_models", "Загрузка модели транскрипции...")

        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            processor = WhisperProcessor.from_pretrained(self.whisper_model_id)
            model = WhisperForConditionalGeneration.from_pretrained(
                self.whisper_model_id,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
            )

            self.whisper_model = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                stride_length_s=(4, 2),
                return_timestamps=True,
            )

            self.current_loaded_model = "whisper"
            logger.info("Whisper загружен успешно")
        except Exception as e:
            logger.error(f"Ошибка загрузки Whisper: {str(e)}")
            await self.force_gpu_cleanup()
            raise

    async def load_diarization_model(self):
        """Загрузка модели PyAnnote для диаризации"""
        if (
            self.diarization_pipeline is not None
            and self.current_loaded_model == "diarization"
        ):
            return

        await self.force_gpu_cleanup()
        logger.info(f"Загрузка модели PyAnnote '{self.diarization_model_id}'...")
        self._update_progress(0, "loading_models", "Загрузка модели диаризации...")

        try:
            # Загрузка пайплайна PyAnnote
            # Используем аргумент 'token' для dev-версии библиотеки
            pipeline = Pipeline.from_pretrained(
                self.diarization_model_id,
                token=self.hf_token,
                cache_dir=self.pyannote_cache_path,
            )

            # Перенос на GPU
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
                logger.info("PyAnnote перемещен на GPU")

            self.diarization_pipeline = pipeline
            self.current_loaded_model = "diarization"
            logger.info("PyAnnote загружен успешно")

        except Exception as e:
            logger.error(f"Ошибка загрузки PyAnnote: {str(e)}")
            logger.error(
                "Убедитесь, что HF_TOKEN верен и вы приняли условия использования модели на HuggingFace"
            )
            await self.force_gpu_cleanup()
            raise

    async def load_qwen_model(self):
        """Загрузка модели Qwen с 4-битным квантованием"""
        if self.qwen_model is not None and self.current_loaded_model == "qwen":
            return

        await self.force_gpu_cleanup()
        logger.info("Загрузка модели Qwen с 4-битным квантованием...")
        self._update_progress(0, "loading_models", "Загрузка модели суммаризации...")

        try:
            if not os.path.exists(self.qwen_model_path):
                raise FileNotFoundError(
                    f"Модель Qwen не найдена по пути: {self.qwen_model_path}"
                )

            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.qwen_model_path, trust_remote_code=True, local_files_only=True
            )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                self.qwen_model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float16,
            )

            self.current_loaded_model = "qwen"
            logger.info("Qwen загружен успешно")
        except Exception as e:
            logger.error(f"Ошибка загрузки Qwen: {str(e)}")
            await self.force_gpu_cleanup()
            raise

    async def unload_current_model(self):
        """Выгрузка текущей модели из памяти GPU"""
        if self.current_loaded_model is None:
            return

        logger.info(
            f"Выгрузка текущей модели ({self.current_loaded_model}) из памяти..."
        )

        if self.current_loaded_model == "whisper":
            # Явно удаляем модель и пайплайн transformers
            del self.whisper_model
            self.whisper_model = None
        elif self.current_loaded_model == "diarization":
            del self.diarization_pipeline
            self.diarization_pipeline = None
        elif self.current_loaded_model == "qwen":
            del self.qwen_model
            del self.qwen_tokenizer
            self.qwen_model = None
            self.qwen_tokenizer = None

        self.current_loaded_model = None
        await self.force_gpu_cleanup()

    # ================= СУММАРИЗАЦИЯ =================

    def smart_split_text(self, text: str, max_chars: int = 14000) -> List[str]:
        """Разбиение текста на части для обработки"""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if sentence_len > max_chars:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                for i in range(0, sentence_len, max_chars):
                    chunks.append(sentence[i : i + max_chars])
                continue

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

    async def fast_summarize_chunk(self, text: str, max_length: int = 800) -> str:
        """Сверхбыстрая суммаризация одного чанка"""
        system_message = """Ты - секретарь, оформляющий протокол видео на русском языке. Используй следующий шаблон:
Шаблон:
Общая тематика видео:
Основные темы:
Ключевые тезисы:
Итоги:

Требования:
1. Строго придерживайся структуры шаблона
2. Выводи только пункты из шаблона
3. Абсолютная грамотность
4. Использовать только русский язык
"""
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Проанализируй текст и создай содержание:\n{text}",
            },
        ]

        prompt = self.qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.qwen_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=12000
        ).to(self.qwen_model.device)

        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
            )

        generated_ids = outputs[0][len(inputs.input_ids[0]) :].tolist()
        summary = self.qwen_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        return summary

    async def fast_summarize_chunk_protocol(
        self, text: str, max_length: int = 1200
    ) -> str:
        """Суммаризация чанка по шаблону протокола"""
        system_message = """Ты - секретарь, оформляющий протокол конференции. Шаблон:
Дата проведения:
Присутствуют:
Повестка дня:
Рассмотрены вопросы и решения:
Итоги:

Требования:
1. Строго придерживайся шаблона
2. Только русский язык
"""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Создай протокол по тексту:\n{text}"},
        ]

        prompt = self.qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.qwen_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=12000
        ).to(self.qwen_model.device)

        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                repetition_penalty=1.05,
            )

        generated_ids = outputs[0][len(inputs.input_ids[0]) :].tolist()
        summary = self.qwen_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        return summary

    async def final_merge_summaries(
        self, combined_text: str, max_length: int = 7000, summary_type: str = "standard"
    ) -> str:
        """Финальное объединение"""
        if summary_type == "standard":
            system_message = "Объедини суммаризации в ЕДИНЫЙ структурированный документ. Убери дубликаты."
        else:
            system_message = "Объедини протоколы в ЕДИНЫЙ протокол конференции. Объедини списки присутствующих и вопросы."

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Объедини текст:\n{combined_text}"},
        ]

        prompt = self.qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.qwen_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=24000
        ).to(self.qwen_model.device)

        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                repetition_penalty=1.05,
            )

        generated_ids = outputs[0][len(inputs.input_ids[0]) :].tolist()
        summary = self.qwen_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        return summary

    async def summarize_long_text_optimized(
        self, text: str, max_length: int = 4000, summary_type: str = "standard"
    ) -> str:
        """Оркестратор суммаризации"""
        logger.info(f"Запуск умной суммаризации ({summary_type}). Длина: {len(text)}")
        await self.load_qwen_model()

        try:
            chunk_summarizer = (
                self.fast_summarize_chunk
                if summary_type == "standard"
                else self.fast_summarize_chunk_protocol
            )

            chunks = self.smart_split_text(text, max_chars=14000)
            part_summaries = []
            total_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                self._update_progress(
                    85 + int(10 / total_chunks * i),
                    "summarization",
                    f"Суммаризация части {i+1}/{total_chunks}",
                )
                summary = await chunk_summarizer(chunk, max_length=1000)
                part_summaries.append(summary)
                await self.light_memory_cleanup()

            if len(part_summaries) == 1:
                return part_summaries[0]

            logger.info("Финальная сборка документа")
            combined_text = "\n\n".join(part_summaries)
            final_summary = await self.final_merge_summaries(
                combined_text, max_length=6000, summary_type=summary_type
            )
            return final_summary

        except Exception as e:
            logger.error(f"Ошибка при умной суммаризации: {str(e)}")
            raise
        finally:
            await self.unload_current_model()

    async def summarize_text(
        self, text: str, max_length: int = 4000, summary_type: str = "standard"
    ) -> str:
        if not text.strip():
            raise ValueError("Текст пуст")
        return await self.summarize_long_text_optimized(text, max_length, summary_type)

    # ================= ВСПОМОГАТЕЛЬНЫЕ =================

    def save_formatted_text(
        self, file_path: str, content: str, file_type: str, is_markdown: bool = False
    ):
        if is_markdown and not file_path.lower().endswith(".md"):
            file_path = file_path.rsplit(".", 1)[0] + ".md"
        try:
            with open(file_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            logger.info(f"Файл сохранен: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения {file_path}: {e}")
            raise

    async def download_from_url(self, url: str, task_id: str) -> str:
        logger.info(f"Скачивание: {url}")
        self._update_progress(10, "download", f"Скачивание: {url}")
        ydl_opts = {"outtmpl": f"uploads/{task_id}.%(ext)s", "ignoreerrors": True}

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            for file in os.listdir("uploads"):
                if file.startswith(task_id) and not file.endswith(".part"):
                    return f"uploads/{file}"
            raise Exception("Файл не найден после скачивания")
        except Exception as e:
            logger.error(f"Ошибка скачивания: {e}")
            raise

    def extract_audio(self, video_path: str, task_id: str) -> str:
        logger.info(f"Извлечение аудио: {video_path}")
        audio_path = f"uploads/{task_id}.wav"
        try:
            # Оставляем принудительную конвертацию в 16kHz mono,
            # чтобы снять нагрузку с PyAnnote и гарантировать совместимость
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-vn",
                audio_path,
                "-y",
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return audio_path
        except Exception as e:
            logger.error(f"Ошибка ffmpeg: {e}")
            raise

    def process_audio_file(self, audio_path: str, task_id: str) -> str:
        logger.info(f"Обработка аудио: {audio_path}")
        converted_path = f"uploads/{task_id}.wav"
        try:
            cmd = [
                "ffmpeg",
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-vn",
                converted_path,
                "-y",
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return converted_path
        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")
            raise

    # ================= CORE LOGIC =================

    def diarize_audio(self, audio_path: str):
        """
        Запуск диаризации через PyAnnote Community-1.
        Использует загрузку в память для избежания конфликтов бэкендов и SegFaults.
        """
        logger.info(f"Запуск PyAnnote Community-1 для: {audio_path}")

        try:
            # 1. Загружаем аудио в память через torchaudio (безопасно)
            # Мы используем backend 'soundfile', который был установлен при инициализации
            waveform, sample_rate = torchaudio.load(audio_path)

            # PyAnnote Community-1 требует 16kHz. Если вдруг ffmpeg не сработал или файл другой:
            if sample_rate != 16000:
                logger.info(f"Resampling audio from {sample_rate} to 16000 Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                waveform = resampler(waveform)
                sample_rate = 16000

            # Если стерео, конвертируем в моно (усреднение каналов)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            logger.info(f"Аудио загружено в память: shape={waveform.shape}")

            # 2. Перемещаем тензор на GPU, если пайплайн там
            # Примечание: PyAnnote pipeline.__call__ обычно ожидает dict на CPU или path,
            # но для безопасности можно передать тензор на том же девайсе.
            # Однако, документация Community-1 рекомендует:
            # output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
            # Тензор waveform должен быть на GPU, если pipeline на GPU.

            if torch.cuda.is_available():
                waveform = waveform.to("cuda")

            # 3. Запуск пайплайна с данными в памяти
            inputs = {"waveform": waveform, "sample_rate": sample_rate}
            output = self.diarization_pipeline(inputs)

            # 4. Извлекаем результат
            diarization = output.speaker_diarization

            final_result = []
            for turn, speaker in diarization:
                final_result.append(
                    {"start": turn.start, "end": turn.end, "speaker": speaker}
                )

            logger.info(f"Диаризация завершена. Сегментов: {len(final_result)}")

            # Очистка тензора
            del waveform
            del inputs

            return final_result

        except Exception as e:
            logger.error(f"Ошибка при выполнении диаризации: {e}")
            raise

    def align_asr_and_diarization(self, asr_segments, diar_segments):
        """Объединение транскрипции и диаризации по времени"""
        aligned = []
        for asr in asr_segments:
            asr_start = asr.get("start")
            asr_end = asr.get("end")

            if asr_start is None or asr_end is None:
                continue

            # Поиск пересечений
            overlap_speakers = {}
            for dia in diar_segments:
                overlap_start = max(asr_start, dia["start"])
                overlap_end = min(asr_end, dia["end"])
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > 0:
                    overlap_speakers[dia["speaker"]] = (
                        overlap_speakers.get(dia["speaker"], 0) + overlap_duration
                    )

            if overlap_speakers:
                dominant_speaker = max(overlap_speakers, key=overlap_speakers.get)
            else:
                dominant_speaker = "UNKNOWN"

            aligned.append(
                {
                    "start": asr_start,
                    "end": asr_end,
                    "speaker": dominant_speaker,
                    "text": asr["text"],
                }
            )
        return aligned

    async def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """Транскрипция через Whisper"""
        logger.info(f"Транскрипция: {audio_path}")
        await self.load_whisper_model()

        try:
            result = self.whisper_model(audio_path)
            segments = []
            if isinstance(result["chunks"], list):
                for chunk in result["chunks"]:
                    segments.append(
                        {
                            "start": chunk["timestamp"][0],
                            "end": chunk["timestamp"][1],
                            "text": chunk["text"].strip(),
                        }
                    )
            else:
                segments.append({"start": 0.0, "end": None, "text": result["text"]})

            return segments
        finally:
            await self.unload_current_model()

    async def process_media(
        self,
        file_path: str,
        task_id: str,
        media_type: str = "video",
        summary_type: str = "standard",
    ) -> Dict:
        """Главный пайплайн обработки"""
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

            # 2. Транскрипция
            self._update_progress(30, "transcription", "Транскрипция речи...")
            transcription = await self.transcribe_audio(audio_path)

            # 3. Диаризация (PyAnnote Community-1)
            self._update_progress(
                60, "diarization", "Определение спикеров (PyAnnote Community-1)..."
            )
            await self.load_diarization_model()
            diarization = self.diarize_audio(audio_path)
            await self.unload_current_model()

            # 4. Сведение
            self._update_progress(75, "merging", "Сведение результатов...")
            merged = self.align_asr_and_diarization(transcription, diarization)

            full_text = " ".join([seg["text"] for seg in merged])
            speakers_count = len(set([seg["speaker"] for seg in merged]))

            # 5. Суммаризация
            self._update_progress(
                85, "summarization", "Создание краткого содержания..."
            )
            summary = await self.summarize_text(full_text, summary_type=summary_type)

            # 6. Форматирование и сохранение
            formatted_transcription = self.format_transcription(merged)
            processing_time = (datetime.now() - start_time).total_seconds() / 60

            formatted_transcription = self.add_processing_metadata(
                formatted_transcription, task_id, processing_time, media_type
            )

            self.save_formatted_text(
                f"results/{task_id}_transcription.md",
                formatted_transcription,
                "Транскрипция",
                is_markdown=True,
            )
            self.save_formatted_text(
                f"results/{task_id}_summary.md",
                summary,
                "Краткое содержание",
                is_markdown=True,
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
            }

        except Exception as e:
            logger.error(f"CRITICAL ERROR: {e}", exc_info=True)
            self._update_progress(0, "error", str(e))
            raise
        finally:
            await self.unload_current_model()
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            await self.force_gpu_cleanup()

    def format_transcription(self, merged_data: List[Dict]) -> str:
        """Форматирование результата для чтения"""
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
        self, content: str, task_id: str, processing_time: float, media_type: str
    ) -> str:
        tomsk_time = self._get_tomsk_time()
        meta = f"""# Отчет обработки
ID: {task_id}
Дата: {tomsk_time.strftime('%Y-%m-%d %H:%M:%S')}
Время обработки: {processing_time:.2f} мин
----------------------------------------
\n"""
        return meta + content

    def check_gpu(self) -> bool:
        return torch.cuda.is_available()
