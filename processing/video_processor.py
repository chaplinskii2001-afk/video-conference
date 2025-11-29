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
from speechbrain.inference.speaker import SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, task_id: str = None):
        self.task_id = task_id
        self.hf_token = os.getenv("HF_TOKEN")
        self.whisper_model = None
        self.speaker_encoder = None
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.current_loaded_model = None
        self.whisper_model_path = "/app/models/whisper"
        self.speechbrain_model_path = "/app/models/speechbrain"
        self.qwen_model_path = "/app/models/qwen"
        self.whisper_model_id = "bond005/whisper-podlodka-turbo"
        self.speechbrain_model_id = "speechbrain/spkrec-ecapa-voxceleb"
        logger.info(
            f"Инициализация VideoProcessor с новыми моделями: Whisper='{self.whisper_model_id}', SpeechBrain='{self.speechbrain_model_id}'"
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

    def get_system_memory_info(self):
        """Получить информацию о системной памяти"""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / 1024**3,
                "available_gb": memory.available / 1024**3,
                "used_gb": memory.used / 1024**3,
                "usage_percent": memory.percent,
            }
        except Exception as e:
            logger.warning(f"Не удалось получить информацию о системной памяти: {e}")
            return {"total_gb": 0, "used_gb": 0, "available_gb": 0, "usage_percent": 0}

    async def force_gpu_cleanup(self):
        """Очистка памяти GPU"""
        logger.info("Очистка памяти GPU...")
        try:
            # Собираем мусор
            gc.collect()
            # Очищаем кэш CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # Даем время на очистку
            await asyncio.sleep(1)
            memory_info = self.get_gpu_memory_info()
            logger.info(
                f"Память после очистки: {memory_info['used_gb']:.2f}/{memory_info['total_gb']:.2f} GB ({memory_info['usage_percent']:.1f}%)"
            )
        except Exception as e:
            logger.warning(f"Ошибка при очистке GPU памяти: {e}")

    async def load_whisper_model(self):
        """Загрузка модели Whisper с 8-битным квантованием"""
        if self.whisper_model is not None and self.current_loaded_model == "whisper":
            return

        # Принудительно очищаем память перед загрузкой
        await self.force_gpu_cleanup()
        logger.info(
            f"Загрузка модели Whisper '{self.whisper_model_id}' с 8-битным квантованием..."
        )
        self._update_progress(0, "loading_models", "Загрузка модели транскрипции...")
        gpu_before = self.get_gpu_memory_info()

        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            logger.info(
                f"device_map='auto' будет использоваться для распределения модели"
            )

            # Загрузка процессора и модели с 8-битным квантованием
            processor = WhisperProcessor.from_pretrained(self.whisper_model_id)
            model = WhisperForConditionalGeneration.from_pretrained(
                self.whisper_model_id,
                load_in_8bit=True,  # Ключевая строка для 8-битного режима
                device_map="auto",  # Автоматическое распределение по GPU/CPU
                torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
            )

            # Создание пайплайна
            self.whisper_model = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,  # Обработка по чанкам (важно для длинных аудио)
                stride_length_s=(4, 2),  # Перекрытие чанков для плавности
                return_timestamps=True,  # Возвращает временные метки по словам/сегментам
            )

            self.current_loaded_model = "whisper"
            gpu_after = self.get_gpu_memory_info()
            memory_used = gpu_after["used_gb"] - gpu_before["used_gb"]
            logger.info(
                f"Whisper '{self.whisper_model_id}' загружен, использовано GPU: {memory_used:.2f} GB"
            )
            logger.info(
                f"GPU после Whisper: {gpu_after['used_gb']:.2f}/{gpu_after['total_gb']:.2f} GB"
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки Whisper '{self.whisper_model_id}': {str(e)}")
            await self.force_gpu_cleanup()
            raise

    async def load_speechbrain_model(self):
        """Загрузка модели SpeechBrain для диаризации"""
        if (
            self.speaker_encoder is not None
            and self.current_loaded_model == "diarization"
        ):
            return

        # Принудительно очищаем память перед загрузкой
        await self.force_gpu_cleanup()
        logger.info(f"Загрузка модели SpeechBrain '{self.speechbrain_model_id}'...")
        self._update_progress(0, "loading_models", "Загрузка модели диаризации...")
        gpu_before = self.get_gpu_memory_info()

        try:
            # Загрузка модели через новый интерфейс SpeechBrain 1.0+
            self.speaker_encoder = SpeakerRecognition.from_hparams(
                source=self.speechbrain_model_id,
                savedir=self.speechbrain_model_path,
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            )
            self.current_loaded_model = "diarization"
            gpu_after = self.get_gpu_memory_info()
            memory_used = gpu_after["used_gb"] - gpu_before["used_gb"]
            logger.info(
                f"SpeechBrain '{self.speechbrain_model_id}' загружен, использовано GPU: {memory_used:.2f} GB"
            )
            logger.info(
                f"GPU после SpeechBrain: {gpu_after['used_gb']:.2f}/{gpu_after['total_gb']:.2f} GB"
            )
        except Exception as e:
            logger.error(
                f"Ошибка загрузки SpeechBrain '{self.speechbrain_model_id}': {str(e)}"
            )
            await self.force_gpu_cleanup()
            raise

    async def load_qwen_model(self):
        """Загрузка модели Qwen с 4-битным квантованием"""
        if self.qwen_model is not None and self.current_loaded_model == "qwen":
            return

        # Принудительно очищаем память перед загрузкой
        await self.force_gpu_cleanup()
        logger.info("Загрузка модели Qwen с 4-битным квантованием...")
        self._update_progress(0, "loading_models", "Загрузка модели суммаризации...")
        gpu_before = self.get_gpu_memory_info()

        try:
            # Проверяем существование модели
            if not os.path.exists(self.qwen_model_path):
                raise FileNotFoundError(
                    f"Модель Qwen не найдена по пути: {self.qwen_model_path}"
                )
            config_path = os.path.join(self.qwen_model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config.json не найден по пути: {config_path}")

            logger.info(f"Загрузка токенизатора из {self.qwen_model_path}")
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.qwen_model_path, trust_remote_code=True, local_files_only=True
            )

            # Конфигурация для 4-битного квантования
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            logger.info(
                f"Загрузка модели с 4-битным квантованием из {self.qwen_model_path}"
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
            gpu_after = self.get_gpu_memory_info()
            memory_used = gpu_after["used_gb"] - gpu_before["used_gb"]
            logger.info(
                f"Qwen с 4-битным квантованием загружен, использовано GPU: {memory_used:.2f} GB"
            )
            logger.info(
                f"GPU после Qwen: {gpu_after['used_gb']:.2f}/{gpu_after['total_gb']:.2f} GB"
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки Qwen: {str(e)}")
            # Показываем содержимое директории для диагностики
            try:
                if os.path.exists(self.qwen_model_path):
                    files = os.listdir(self.qwen_model_path)
                    logger.error(
                        f"Содержимое директории {self.qwen_model_path}: {files}"
                    )
            except Exception as dir_error:
                logger.error(f"Не удалось прочитать директорию: {dir_error}")
            await self.force_gpu_cleanup()
            raise

    async def unload_current_model(self):
        """Выгрузка текущей модели из памяти GPU"""
        if self.current_loaded_model is None:
            return

        logger.info(
            f"Выгрузка текущей модели ({self.current_loaded_model}) из памяти..."
        )

        if self.current_loaded_model == "whisper" and self.whisper_model is not None:
            try:
                del self.whisper_model
                self.whisper_model = None
            except Exception as e:
                logger.warning(f"Ошибка при выгрузке Whisper: {e}")
        elif (
            self.current_loaded_model == "diarization"
            and self.speaker_encoder is not None
        ):
            try:
                del self.speaker_encoder
                self.speaker_encoder = None
            except Exception as e:
                logger.warning(f"Ошибка при выгрузке SpeechBrain: {e}")
        elif self.current_loaded_model == "qwen" and self.qwen_model is not None:
            try:
                del self.qwen_model
                del self.qwen_tokenizer
                self.qwen_model = None
                self.qwen_tokenizer = None
            except Exception as e:
                logger.warning(f"Ошибка при выгрузке Qwen: {e}")

        self.current_loaded_model = None
        # Принудительная очистка памяти
        await self.force_gpu_cleanup()

    async def summarize_long_text_optimized(
        self, text: str, max_length: int = 4000, summary_type: str = "standard"
    ) -> str:
        """Оптимизированная суммаризация с разделением на максимум 2 части"""
        logger.info(
            f"Быстрая суммаризация длинного текста ({summary_type}): {len(text)} символов"
        )

        # Загружаем модель ОДИН РАЗ
        await self.load_qwen_model()
        try:
            # Определяем функцию суммаризации в зависимости от типа
            chunk_summarizer = (
                self.fast_summarize_chunk
                if summary_type == "standard"
                else self.fast_summarize_chunk_protocol
            )

            # Разделяем текст на части (максимум 2 части)
            if len(text) <= 20000:
                logger.info("Текст короткий, обрабатывается целиком")
                # Обрабатываем весь текст сразу
                summary = await chunk_summarizer(text, max_length=max_length)
                return summary
            else:
                logger.info(
                    f"Текст длинный ({len(text)} символов), разделяем на 2 части"
                )
                parts = self.split_text_into_two_parts(text)
                logger.info(f"Текст разделен на {len(parts)} части")

                # Обрабатываем каждую часть отдельно
                part_summaries = []
                for i, part in enumerate(parts):
                    logger.info(
                        f"Быстрая суммаризация части {i+1}/{len(parts)} ({summary_type})"
                    )
                    self._update_progress(
                        85 + int(10 * i / len(parts)),
                        "summarization",
                        f"Суммаризация части {i+1}/{len(parts)} ({summary_type})",
                    )

                    # Используем выбранную функцию суммаризации с увеличенным max_length для частей
                    part_max_length = min(2000, max_length // 2)
                    summary = await chunk_summarizer(part, max_length=part_max_length)
                    part_summaries.append(summary)

                    # Легкая очистка между частями
                    await self.light_memory_cleanup()

                # Финальная суммаризация объединенных результатов
                logger.info("Финальная суммаризация объединенных частей")
                self._update_progress(
                    95, "summarization", "Финальная суммаризация объединенных частей"
                )

                combined_text = "\n\n".join(part_summaries)

                # Для финальной суммаризации используем специальный промпт для объединения
                final_summary = await self.final_merge_summaries(
                    combined_text, max_length=max_length, summary_type=summary_type
                )
                return final_summary

        except Exception as e:
            logger.error(f"Ошибка при быстрой суммаризации ({summary_type}): {str(e)}")
            raise
        finally:
            # Выгружаем модель только один раз в конце
            await self.unload_current_model()

    def split_text_into_two_parts(self, text: str) -> List[str]:
        """Разделение текста на 2 равные части по предложениям"""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        total_sentences = len(sentences)

        if total_sentences <= 1:
            return [text]

        # Находим середину по количеству символов, а не предложений
        total_length = len(text)
        target_length = total_length // 2

        # Ищем оптимальное место раздела
        current_length = 0
        split_index = 0

        for i, sentence in enumerate(sentences):
            current_length += len(sentence)
            if current_length >= target_length:
                split_index = i
                break

        # Создаем две части
        part1 = " ".join(sentences[: split_index + 1])
        part2 = " ".join(sentences[split_index + 1 :])

        logger.info(
            f"Разделение текста: часть 1 - {len(part1)} символов, часть 2 - {len(part2)} символов"
        )
        return [part1, part2]

    async def light_memory_cleanup(self):
        """Легкая очистка памяти без выгрузки модели"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        await asyncio.sleep(0.1)

    async def fast_summarize_chunk(self, text: str, max_length: int = 800) -> str:
        """Сверхбыстрая суммаризация одного чанка"""
        logger.info(f"Быстрая суммаризация чанка: {len(text)} символов")
        # Минималистичный промпт для скорости
        system_message = """Ты - секретарь, оформляющий протокол видео на русском языке. Используй следующий шаблон:
Шаблон:
Общая тематика видео:
Основные темы:
Ключевые тезисы:
Итоги:

Требования:
1. Строго придерживайся структуры шаблона
2. Выводи только пункты из шаблона, без дополнительных вводных слов и заголовков
3. Абсолютная грамотность
4. Использовать только русский язык
5. Не упускать детали
"""

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Проанализируй текст и создай строгое структурированное содержание:\n{text}",
            },
        ]

        prompt = self.qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        # Жесткое ограничение длины входа
        inputs = self.qwen_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=12000,  # Увеличил для работы с большими текстами
        ).to(self.qwen_model.device)

        # Оптимизированные настройки для скорости и экономии памяти
        generation_config = {
            "max_new_tokens": max_length,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.05,
            "early_stopping": False,
            "pad_token_id": self.qwen_tokenizer.eos_token_id,
        }

        # Отключаем градиенты и включаем экономию памяти
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.qwen_model.generate(**inputs, **generation_config)

        generated_ids = outputs[0][len(inputs.input_ids[0]) :].tolist()
        summary = self.qwen_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        logger.info(f"Быстрая суммаризация завершена: {len(summary)} символов")
        return summary

    async def fast_summarize_chunk_protocol(
        self, text: str, max_length: int = 1200
    ) -> str:
        """Сверхбыстрая суммаризация одного чанка по шаблону протокола"""
        logger.info(
            f"Быстрая суммаризация чанка по шаблону протокола: {len(text)} символов"
        )
        # Минималистичный промпт для скорости
        system_message = """Ты - секретарь, оформляющий протокол конференции на русском языке. Используй следующий шаблон:
Шаблон:
Дата проведения конференции: (Если была упомянута)
Место проведения - (название организации/предприятия)
Присутствуют: (количество человек/фамилии)
Повестка дня: (Список вопросов, которые обсуждались на конференции, кратко по пунктам)
Рассмотрены вопросы и прияты решения: (по пунктам, какие вопросы из повестки были рассмотрены и какие решения приняты, уже детально и подробно)
Итоги: (краткое резюмирование всех принятых решений)

Требования:
1. Строго придерживайся структуры шаблона
2. Выводи только пункты из шаблона, без дополнительных вводных слов и заголовков
3. Абсолютная грамотность
4. Использовать только русский язык
5. Не упускать детали
"""

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Создай протокол конференции по следующему тексту:\n{text}",
            },
        ]

        prompt = self.qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        # Жесткое ограничение длины входа
        inputs = self.qwen_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=12000,  # Увеличил для работы с большими текстами
        ).to(self.qwen_model.device)

        # Оптимизированные настройки для скорости и экономии памяти
        generation_config = {
            "max_new_tokens": max_length,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.05,
            "early_stopping": False,
            "pad_token_id": self.qwen_tokenizer.eos_token_id,
        }

        # Отключаем градиенты и включаем экономию памяти
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.qwen_model.generate(**inputs, **generation_config)

        generated_ids = outputs[0][len(inputs.input_ids[0]) :].tolist()
        summary = self.qwen_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        logger.info(f"Быстрая суммаризация завершена: {len(summary)} символов")
        return summary

    async def final_merge_summaries(
        self, combined_text: str, max_length: int = 4000, summary_type: str = "standard"
    ) -> str:
        """Финальное объединение суммаризаций в единый документ"""
        logger.info(
            f"Финальное объединение суммаризаций ({summary_type}): {len(combined_text)} символов"
        )

        if summary_type == "standard":
            system_message = """Ты - секретарь, оформляющий финальное краткое содержание видео на русском языке.
Тебе предоставлены несколько суммаризаций одного и того же видео. Объедини их в ЕДИНОЕ структурированное содержание.

Шаблон:
Общая тематика видео:
Основные темы:
Ключевые тезисы:
Итоги:

ВАЖНЫЕ ТРЕБОВАНИЯ:
1. Строго придерживайся структуры шаблона
2. Создай ЕДИНЫЙ документ, а не несколько отдельных
3. Убери все повторения и дубликаты
4. Объедини информацию из всех частей в логичную структуру
5. Выводи только пункты из шаблона, без дополнительных вводных слов
6. Абсолютная грамотность
7. Использовать только русский язык
"""
        else:  # protocol
            system_message = """Ты - секретарь, оформляющий финальный протокол конференции на русском языке.
Тебе предоставлены несколько протоколов одной и той же конференции. Объедини их в ЕДИНЫЙ протокол.

Шаблон:
Дата проведения конференции: (укажи единую дату, если возможно)
Место проведения - (укажи единое место проведения)
Присутствуют: (объедини список присутствующих)
Повестка дня: (объедини все вопросы в единый список)
Рассмотрены вопросы и прияты решения: (объедини всю информацию по каждому вопросу)
Итоги: (краткое резюмирование всех принятых решений)

ВАЖНЫЕ ТРЕБОВАНИЯ:
1. Строго придерживайся структуры шаблона
2. Создай ЕДИНЫЙ протокол, а не несколько отдельных
3. Убери все повторения дат, мест проведения, списков присутствующих
4. Объедини информацию из всех частей в логичную структуру
5. Выводи только пункты из шаблона, без дополнительных вводных слов
6. Абсолютная грамотность
7. Использовать только русский язык
"""

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Объедини следующие суммаризации в ЕДИНЫЙ документ:\n{combined_text}",
            },
        ]

        prompt = self.qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        # Увеличиваем максимальную длину входа для финального объединения
        inputs = self.qwen_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=16000
        ).to(self.qwen_model.device)

        # Настройки для качественной финальной суммаризации
        generation_config = {
            "max_new_tokens": max_length,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.05,
            "early_stopping": False,
            "pad_token_id": self.qwen_tokenizer.eos_token_id,
        }

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.qwen_model.generate(**inputs, **generation_config)

        generated_ids = outputs[0][len(inputs.input_ids[0]) :].tolist()
        final_summary = self.qwen_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        logger.info(f"Финальное объединение завершено: {len(final_summary)} символов")
        return final_summary

    async def summarize_text(
        self, text: str, max_length: int = 4000, summary_type: str = "standard"
    ) -> str:
        """Переработанная суммаризация с максимальной скоростью"""
        logger.info(f"Ускоренная суммаризация ({summary_type}): {len(text)} символов")
        if not text.strip():
            raise ValueError("Текст для суммаризации пуст")

        # Используем оптимизированный подход для всех случаев
        return await self.summarize_long_text_optimized(text, max_length, summary_type)

    def _improve_russian_text_quality(self, text: str) -> str:
        """Дополнительная обработка для улучшения качества русского текста"""
        if not text:
            return text

        # Исправляем распространенные ошибки
        replacements = {
            # Исправляем смешение раскладок
            "y": "у",
            "e": "е",
            "x": "х",
            "a": "а",
            "o": "о",
            "c": "с",
            "p": "р",
            "k": "к",
            "n": "н",
            "m": "м",
            "t": "т",
            "b": "б",
            "h": "н",
            "r": "р",
            "u": "и",
            "ё": "е",  # В некоторых случаях заменяем ё на е для единообразия
            # Исправляем частые орфографические ошибки
            "щщ": "щ",
            "zz": "зз",
            "aa": "аа",
            "оо": "о",
            # Убираем лишние пробелы
            "  ": " ",
            " .": ".",
            " ,": ",",
            " :": ":",
        }

        # Применяем замены
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Убираем множественные переносы строк
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Проверяем, что заголовки правильно оформлены
        lines = text.split("\n")
        improved_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                improved_lines.append("")
                continue
            # Проверяем заголовки
            if any(
                marker in line
                for marker in [
                    "ОСНОВНЫЕ ТЕМЫ",
                    "КЛЮЧЕВЫЕ ТЕЗИСЫ",
                    "ВЫВОДЫ",
                    "ПРЕДЛОЖЕНИЯ",
                ]
            ):
                # Убедимся, что заголовок правильно оформлен
                if not line.endswith(":"):
                    line = line + ":"
            improved_lines.append(line)

        text = "\n".join(improved_lines)
        return text.strip()

    def save_formatted_text(
        self, file_path: str, content: str, file_type: str, is_markdown: bool = False
    ):
        """Сохранение текста с правильной кодировкой и форматированием"""
        # Если is_markdown True, меняем расширение на .md
        if is_markdown:
            if not file_path.lower().endswith(".md"):
                file_path = file_path.rsplit(".", 1)[0] + ".md"
        try:
            with open(file_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            # В логе указываем реальное расширение файла
            actual_ext = "Markdown" if is_markdown else "Text"
            logger.info(f"{actual_ext} файл успешно сохранен: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения {file_path}: {str(e)}")
            raise

    async def download_from_url(self, url: str, task_id: str) -> str:
        logger.info(f"Скачивание медиа по URL: {url}")
        self._update_progress(10, "download", f"Скачивание медиа по URL: {url}")

        ydl_opts = {
            "outtmpl": f"uploads/{task_id}.%(ext)s",
            "socket_timeout": 60,
            "retries": 15,
            "fragment_retries": 15,
            "extract_flat": False,
            "ignoreerrors": True,
            "no_warnings": False,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-us,en;q=0.5",
                "Accept-Encoding": "gzip,deflate",
                "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.7",
                "Connection": "keep-alive",
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise Exception(f"Не удалось получить информацию о медиа: {url}")
                logger.info(f"Найдено медиа: {info.get('title', 'Unknown')}")
                logger.info(f"Тип: {info.get('extractor', 'Unknown')}")
                if "duration" in info:
                    logger.info(f"Длительность: {info.get('duration')} сек")
                ydl.download([url])

            for file in os.listdir("uploads"):
                if file.startswith(task_id) and not file.endswith(".part"):
                    file_path = f"uploads/{file}"
                    if os.path.getsize(file_path) < 1024:
                        with open(file_path, "r") as f:
                            content = f.read(500)
                            if (
                                "<html" in content.lower()
                                or "<!doctype" in content.lower()
                            ):
                                os.remove(file_path)
                                raise Exception(
                                    f"Скачанный файл является HTML страницей, а не медиа"
                                )
                    logger.info(
                        f"Медиа успешно скачано: {file_path}, размер: {os.path.getsize(file_path)} байт"
                    )
                    self._update_progress(15, "download", "Медиа успешно скачано")
                    return file_path

            raise Exception("Файл не был скачан или не найден в директории uploads")
        except Exception as e:
            logger.error(f"Ошибка скачивания медиа: {str(e)}", exc_info=True)
            logger.error(f"URL который вызвал ошибку: {url}")
            raise

    def extract_audio(self, video_path: str, task_id: str) -> str:
        logger.info(f"Извлечение аудио из: {video_path}")
        self._update_progress(20, "audio_extraction", "Извлечение аудио из медиа")
        audio_path = f"uploads/{task_id}.wav"

        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Файл не найден: {video_path}")
            file_size = os.path.getsize(video_path)
            if file_size < 1024:
                raise Exception(f"Файл слишком мал для обработки: {file_size} байт")
            logger.info(f"Размер видео файла: {file_size} байт")

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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"FFmpeg stderr: {result.stderr}")
                logger.error(f"FFmpeg stdout: {result.stdout}")
                raise Exception(f"FFmpeg error: {result.stderr}")

            if not os.path.exists(audio_path):
                raise Exception(f"Аудио файл не создан: {audio_path}")
            audio_size = os.path.getsize(audio_path)
            logger.info(f"Аудио успешно извлечено, размер: {audio_size} байт")
            self._update_progress(25, "audio_extraction", "Аудио успешно извлечено")
            return audio_path
        except Exception as e:
            logger.error(f"Ошибка извлечения аудио: {str(e)}", exc_info=True)
            raise

    def get_speech_segments(self, audio_path: str, sampling_rate=16000):
        """
        Возвращает список сегментов с активной речью: [{"start": float, "end": float}, ...]
        """
        model_vad = load_silero_vad()
        wav = read_audio(audio_path, sampling_rate=sampling_rate)
        speech_timestamps = get_speech_timestamps(
            wav,
            model_vad,
            sampling_rate=sampling_rate,
            threshold=0.5,  # Порог чувствительности
        )

        # Преобразуем в секунды
        segments = []
        for ts in speech_timestamps:
            segments.append(
                {"start": ts["start"] / sampling_rate, "end": ts["end"] / sampling_rate}
            )
        return segments

    def extract_embeddings(
        self, audio_path: str, speech_segments: list, sampling_rate=16000
    ):
        """
        Возвращает: embeddings (np.array), segments (с обновлёнными границами)
        """
        # Загрузка всего аудио
        waveform, sr = torchaudio.load(audio_path)
        if sr != sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)

        embeddings = []
        valid_segments = []

        for seg in speech_segments:
            start_sample = int(seg["start"] * sampling_rate)
            end_sample = int(seg["end"] * sampling_rate)
            # Извлечение сегмента
            segment_wave = waveform[:, start_sample:end_sample]
            # Пропуск слишком коротких сегментов (<0.5 сек)
            if segment_wave.size(1) < sampling_rate // 2:
                continue
            try:
                # Извлечение эмбеддинга (нормализованный, 192-мерный)
                # Используем новый интерфейс SpeechBrain 1.0+
                emb = self.speaker_encoder.encode_batch(segment_wave)
                embeddings.append(emb.squeeze().cpu().numpy())
                valid_segments.append(seg)
            except Exception as e:
                print(f"Ошибка при обработке сегмента {seg}: {e}")
                continue

        return np.array(embeddings), valid_segments

    def cluster_speakers(self, embeddings: np.ndarray, min_clusters=1, max_clusters=10):
        """
        Возвращает метки кластеров для каждого эмбеддинга.
        """
        if len(embeddings) == 0:
            return []
        if len(embeddings) == 1:
            return [0]

        # Вычисление матрицы косинусного сходства
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix  # Преобразуем в расстояние

        # Определяем оптимальное число кластеров
        n_clusters = min(max_clusters, len(embeddings))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
        labels = clustering.fit_predict(distance_matrix)
        return labels.tolist()

    def diarize_audio(self, audio_path: str):
        """
        Возвращает список: [{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]
        """
        # 1. Получаем сегменты с речью
        speech_segs = self.get_speech_segments(audio_path)
        # 2. Извлекаем эмбеддинги
        embeddings, valid_segs = self.extract_embeddings(audio_path, speech_segs)
        # 3. Кластеризуем
        labels = self.cluster_speakers(embeddings)
        # 4. Формируем результат
        diarization = []
        for i, seg in enumerate(valid_segs):
            speaker_id = f"SPEAKER_{labels[i]:02d}" if labels else "SPEAKER_00"
            diarization.append(
                {"start": seg["start"], "end": seg["end"], "speaker": speaker_id}
            )
        return diarization

    def align_asr_and_diarization(self, asr_segments, diar_segments):
        """
        Объединяет результаты транскрипции и диаризации.
        Обрабатывает случаи, когда у ASR-сегмента start или end равны None.
        """
        aligned = []
        for asr in asr_segments:
            # Проверяем, что start и end определены
            asr_start = asr.get("start")
            asr_end = asr.get("end")

            # Пропускаем сегменты, у которых нет хотя бы одного таймстемпа
            if asr_start is None or asr_end is None:
                # Логируем пропуск, если нужно
                # logger.warning(f"Пропуск сегмента ASR без таймстемпа: {asr}")
                continue

            # Найти доминирующего спикера в интервале [asr.start, asr.end]
            overlap_speakers = {}
            for dia in diar_segments:
                # Проверяем, что у диаризации тоже есть таймстемпы (для надежности)
                dia_start = dia.get("start")
                dia_end = dia.get("end")
                if dia_start is None or dia_end is None:
                    continue  # Пропускаем сегмент диаризации без таймстемпа

                # Рассчитываем пересечение
                overlap_start = max(asr_start, dia_start)
                overlap_end = min(asr_end, dia_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > 0:
                    overlap_speakers[dia["speaker"]] = (
                        overlap_speakers.get(dia["speaker"], 0) + overlap_duration
                    )

            # Определяем спикера для сегмента ASR
            if overlap_speakers:
                dominant_speaker = max(overlap_speakers, key=overlap_speakers.get)
            else:
                # Если пересечений не найдено, присваиваем "UNKNOWN" или первый спикер из диаризации
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
        """Транскрипция с Whisper-podlodka-turbo"""
        logger.info(f"Начало транскрипции аудио: {audio_path}")
        self._update_progress(30, "transcription", "Начало транскрипции аудио")

        # Загружаем модель Whisper
        await self.load_whisper_model()

        try:
            self._update_progress(40, "transcription", "Выполнение транскрипции...")
            result = self.whisper_model(audio_path)

            # Результат может быть в двух форматах:
            # 1. Список слов с таймстемпами (если return_timestamps="word")
            # 2. Список сегментов (по умолчанию)
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
                # Fallback: весь текст как один сегмент
                segments.append({"start": 0.0, "end": None, "text": result["text"]})

            logger.info(f"Транскрипция завершена, сегментов: {len(segments)}")
            self._update_progress(
                50,
                "transcription",
                f"Транскрипция завершена: {len(segments)} сегментов",
            )

            if segments:
                sample_segment = segments[0]
                logger.info(
                    f"Пример сегмента: время [{sample_segment['start']:.1f}-{sample_segment['end']:.1f}], текст: {sample_segment['text'][:100]}"
                )

            return segments
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {str(e)}", exc_info=True)
            raise
        finally:
            # Выгружаем Whisper после транскрипции
            await self.unload_current_model()

    async def process_media(
        self,
        file_path: str,
        task_id: str,
        media_type: str = "video",
        summary_type: str = "standard",
    ) -> Dict:
        logger.info(
            f"Начало обработки {media_type} (суммаризация: {summary_type}): {file_path}"
        )
        self._update_progress(
            15, "preprocessing", f"Подготовка к обработке {media_type}"
        )

        audio_path = None
        start_time = datetime.now()

        try:
            await self.force_gpu_cleanup()

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл не найден: {file_path}")
            file_size = os.path.getsize(file_path)
            if file_size < 1024:
                raise Exception(f"Файл слишком мал для обработки: {file_size} байт")

            # Извлечение аудио
            self._update_progress(20, "audio_extraction", "Извлечение аудио из медиа")
            if media_type == "audio":
                audio_path = self.process_audio_file(file_path, task_id)
            else:
                audio_path = self.extract_audio(file_path, task_id)

            # Транскрипция
            self._update_progress(40, "transcription", "Начало транскрипции аудио")
            transcription = await self.transcribe_audio(audio_path)

            # Диаризация
            self._update_progress(60, "diarization", "Начало диаризации")
            await self.load_speechbrain_model()
            diarization = self.diarize_audio(audio_path)
            await self.unload_current_model()

            # Объединение результатов
            self._update_progress(
                70, "merging", "Объединение транскрипции и диаризации"
            )
            merged = self.align_asr_and_diarization(transcription, diarization)

            full_transcription = " ".join([seg["text"] for seg in merged])
            speakers_count = len(set([seg["speaker"] for seg in merged]))
            logger.info(
                f"Обработано сегментов: {len(merged)}, спикеров: {speakers_count}"
            )

            # Суммаризация
            self._update_progress(
                85, "summarization", "Начало суммаризации текста ({summary_type})"
            )
            summary = await self.summarize_text(
                full_transcription, summary_type=summary_type
            )

            formatted_transcription = self.format_transcription(merged)
            processing_time = (datetime.now() - start_time).total_seconds() / 60
            formatted_transcription = self.add_processing_metadata(
                formatted_transcription, task_id, processing_time, media_type
            )

            transcription_file = f"results/{task_id}_transcription.txt"
            summary_file = f"results/{task_id}_summary.txt"

            self.save_formatted_text(
                transcription_file,
                formatted_transcription,
                "Транскрипция",
                is_markdown=True,
            )
            self.save_formatted_text(
                summary_file, summary, "Краткое содержание", is_markdown=True
            )

            self._update_progress(95, "saving", "Сохранение результатов")
            logger.info(f"Обработка завершена для задачи {task_id}")

            result = {
                "task_id": task_id,
                "summary": summary,
                "transcription_length": len(formatted_transcription),
                "segments_count": len(merged),
                "speakers_count": speakers_count,
                "diarization_available": True,
                "processing_time_minutes": round(processing_time, 2),
                "media_type": media_type,
                "summary_type": summary_type,
            }
            return result
        except Exception as e:
            self._update_progress(0, "error", f"Ошибка обработки: {str(e)}")
            logger.error(f"Ошибка обработки {media_type}: {str(e)}", exc_info=True)
            raise
        finally:
            await self.unload_current_model()
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info(f"Временный аудио файл удален: {audio_path}")
                except Exception as e:
                    logger.warning(
                        f"Не удалось удалить временный файл {audio_path}: {e}"
                    )
            await self.force_gpu_cleanup()

    def process_audio_file(self, audio_path: str, task_id: str) -> str:
        """Обработка аудиофайла: конвертация в нужный формат при необходимости"""
        logger.info(f"Обработка аудиофайла: {audio_path}")
        # Проверяем расширение файла
        file_ext = os.path.splitext(audio_path)[1].lower()
        # Если файл уже в формате WAV с нужными параметрами, используем как есть
        if file_ext == ".wav":
            # Проверим параметры аудио
            try:
                probe_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=sample_rate,channels",
                    "-of",
                    "csv=p=0",
                    audio_path,
                ]
                probe_result = subprocess.run(
                    probe_cmd, capture_output=True, text=True, timeout=30
                )
                if probe_result.returncode == 0:
                    sample_rate, channels = probe_result.stdout.strip().split(",")
                    if int(sample_rate) == 16000 and int(channels) == 1:
                        logger.info("Аудиофайл уже в нужном формате (16kHz, mono)")
                        return audio_path
            except Exception as e:
                logger.warning(
                    f"Не удалось проверить параметры аудио, конвертируем: {e}"
                )

        # Конвертируем в нужный формат
        converted_path = f"uploads/{task_id}.wav"
        logger.info(f"Конвертация аудио в WAV (16kHz, mono): {converted_path}")
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            logger.error(f"FFmpeg stdout: {result.stdout}")
            raise Exception(f"Ошибка конвертации аудио: {result.stderr}")

        if not os.path.exists(converted_path):
            raise Exception(f"Сконвертированный аудио файл не создан: {converted_path}")
        audio_size = os.path.getsize(converted_path)
        logger.info(f"Аудио сконвертировано, размер: {audio_size} байт")
        return converted_path

    def format_transcription(self, merged_data: List[Dict]) -> str:
        """Форматирование транскрипции"""
        logger.info("Форматирование транскрипции")
        self._update_progress(80, "formatting", "Форматирование транскрипции")

        formatted = "ТРАНСКРИПЦИЯ ВИДЕО-КОНФЕРЕНЦИИ\n"
        formatted += "=" * 50 + "\n"

        current_speaker = None
        speaker_segments = []

        for segment in merged_data:
            start_min = int(segment["start"] // 60)
            start_sec = int(segment["start"] % 60)
            speaker = segment["speaker"]
            text = segment["text"].strip()
            if not text or len(text) < 2:
                continue

            if speaker != current_speaker and speaker_segments:
                formatted += self._format_speaker_block(
                    current_speaker, speaker_segments
                )
                speaker_segments = []

            current_speaker = speaker
            speaker_segments.append(
                {"time": f"{start_min:02d}:{start_sec:02d}", "text": text}
            )

        if speaker_segments:
            formatted += self._format_speaker_block(current_speaker, speaker_segments)

        return formatted

    def _format_speaker_block(self, speaker: str, segments: List[Dict]) -> str:
        """Форматирование блока одного спикера"""
        block = f"СПИКЕР: {speaker}\n"
        block += "-" * 30 + "\n"
        for segment in segments:
            block += f"[{segment['time']}] {segment['text']}\n"
        block += "\n"
        return block

    def add_processing_metadata(
        self,
        content: str,
        task_id: str,
        processing_time: float,
        media_type: str = "video",
    ) -> str:
        """Добавление метаданных о процессе обработки"""
        tomsk_time = self._get_tomsk_time()
        media_type_text = "аудио" if media_type == "audio" else "видео"
        metadata = f"""ОБРАБОТКА ЗАВЕРШЕНА
Идентификатор задачи: {task_id}
Тип медиа: {media_type_text}
Время обработки: {processing_time:.2f} минут
Дата обработки (Томское время): {tomsk_time.strftime('%Y-%m-%d %H:%M:%S')}
Использована модель транскрипции: Whisper-podlodka-turbo (8-битное квантование)
Использована модель диаризации: SpeechBrain + Agglomerative Clustering
Использована модель суммаризации: Qwen3-4B (4-битное квантование)
"""
        return metadata + content

    def check_gpu(self) -> bool:
        return torch.cuda.is_available()
