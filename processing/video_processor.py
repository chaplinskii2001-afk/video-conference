"""
Основной процессор для обработки видео и аудио файлов
Выполняет транскрипцию, диаризацию и суммаризацию
"""

import os
import re
import asyncio
import subprocess
import logging
import warnings
from typing import Optional
import torch
import torchaudio
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
import yt_dlp
from transformers import StoppingCriteria, StoppingCriteriaList

# Подавляем предупреждения о torchaudio deprecation (будут актуальны в версии 2.9+)
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
# Подавляем экспериментальное предупреждение о chunking в Whisper
warnings.filterwarnings("ignore", message=r".*chunk_length_s is very experimental.*")
# Подавляем предупреждение о forced_decoder_ids
warnings.filterwarnings("ignore", message=r".*forced_decoder_ids.*")
# Подавляем предупреждение о language detection в Whisper
warnings.filterwarnings(
    "ignore", message=r".*multilingual Whisper will default to language detection.*"
)
# Подавляем предупреждение PyAnnote о выключении TF32 (мы включаем TF32 в проекте)
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
)
warnings.filterwarnings(
    "ignore",
    module=r"pyannote\.audio\.utils\.reproducibility",
)

_WORD_CHARS = r"0-9A-Za-zА-Яа-яЁё_"
_LB = rf"(?<![{_WORD_CHARS}])"
_RB = rf"(?![{_WORD_CHARS}])"

_PROFANITY_RE = re.compile(
    rf"{_LB}(?:"
    rf"бля|бляд[а-яё]*|блят[а-яё]*|"
    rf"(?:е|ё)б[а-яё]*|"
    rf"пизд[а-яё]*|"
    rf"ху(?:й|е|ё|я|и|ю)[а-яё]*|"
    rf"сука[а-яё]*|"
    rf"нахуй|нахуя|нахер"
    rf"){_RB}",
    flags=re.IGNORECASE,
)


def remove_profanity_transcription(text: str) -> str:
    """
    Удаляет мат только как отдельные слова/словоформы.
    Подстроки внутри слов не трогает: "потреБЛЯТЬ" останется как есть.
    """
    if not text:
        return text

    def _clean_line(line: str) -> str:
        s = _PROFANITY_RE.sub("", line)
        s = re.sub(r"[ \t]{2,}", " ", s)
        s = re.sub(r"\s+([,.!?;:])", r"\1", s)
        return s

    # сохраняем переносы строк как есть
    lines = text.splitlines(keepends=True)
    out = []
    for ln in lines:
        if ln.endswith("\n"):
            out.append(_clean_line(ln[:-1]) + "\n")
        else:
            out.append(_clean_line(ln))
    return "".join(out)


# Локальные модули
from config import AppConfig, GPUConfig
from processing.gpu_manager import GPUMemoryManager
from processing.model_manager import ModelManager
from processing.audio_enhancer import (
    enhance_audio,
    normalize_loudness,
    split_by_vad,
    filter_transcription_results,
    cleanup_temp_files,
    cleanup_vad_directory,
)


class StopOnTokenSequence(StoppingCriteria):
    def __init__(
        self,
        stop_token_ids: List[int],
        start_length: int,
        max_trailing_tokens: int = 5,
    ):
        self.stop_token_ids = stop_token_ids
        self.start_length = start_length
        self.max_trailing_tokens = max_trailing_tokens

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        if input_ids.shape[0] != 1:
            return False

        if input_ids.shape[1] < self.start_length + len(self.stop_token_ids):
            return False

        window_size = len(self.stop_token_ids) + self.max_trailing_tokens
        tail_window = input_ids[0, -window_size:].tolist()

        for trailing in range(
            0,
            min(self.max_trailing_tokens, len(tail_window) - len(self.stop_token_ids))
            + 1,
        ):
            if trailing == 0:
                candidate = tail_window[-len(self.stop_token_ids) :]
            else:
                candidate = tail_window[
                    -len(self.stop_token_ids) - trailing : -trailing
                ]

            if candidate == self.stop_token_ids:
                return True

        return False


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

    def __init__(
        self,
        task_id: Optional[str] = None,
        *,
        upload_dir: str = "uploads",
        results_dir: str = "results",
    ):
        self.task_id = task_id
        self.upload_dir = upload_dir
        self.results_dir = results_dir

        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Включаем TF32 для лучшей производительности на совместимых GPU
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Инициализация конфигурации
        self.app_config = AppConfig()
        self.gpu_config = AppConfig.get_gpu_config()
        self.model_config = AppConfig.get_model_config()

        # Создаем полную конфигурацию
        self.config = {
            "gpu_config": self.gpu_config,
            "model_config": self.model_config,
            "app_config": self.app_config,
        }

        # Менеджеры
        self.gpu_manager = GPUMemoryManager(log_memory_changes=True)
        self.model_manager = ModelManager(self.config, self.gpu_manager)
        self._assert_models_ready()

        # Настройка CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.logger.info("=" * 60)
        self.logger.info("VideoProcessor инициализирован")
        self.logger.info(f"GPU профиль: {self.gpu_config.get('name', 'Unknown')}")
        self.logger.info(
            f"GPU память: {self.gpu_config.get('gpu_info', {}).get('vram_gb', 0)} GB"
        )
        self.logger.info("=" * 60)

    def _assert_models_ready(self) -> None:
        mc = self.model_config
        required_dirs = [
            mc.get("whisper_path"),
            mc.get("diarization_path"),
            mc.get("qwen_path"),
            os.getenv("XDG_CACHE_HOME", "/app/models/xdg_cache"),
        ]
        for p in required_dirs:
            if not p or not os.path.exists(p):
                raise FileNotFoundError(f"Required dir missing: {p!r}")

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

        ydl_opts = {
            "outtmpl": os.path.join(self.upload_dir, f"{task_id}.%(ext)s"),
            "ignoreerrors": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Находим скачанный файл
            for file in os.listdir(self.upload_dir):
                if file.startswith(task_id) and not file.endswith(".part"):
                    file_path = os.path.join(self.upload_dir, file)
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
        audio_path = os.path.join(self.upload_dir, f"{task_id}.wav")

        try:
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-ac",
                "1",  # моно
                "-ar",
                "16000",  # 16 kHz
                "-vn",  # без видео
                audio_path,
                "-y",  # перезапись
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
        converted_path = os.path.join(self.upload_dir, f"{task_id}.wav")

        try:
            cmd = [
                "ffmpeg",
                "-i",
                audio_path,
                "-ac",
                "1",  # моно
                "-ar",
                "16000",  # 16 kHz
                "-vn",
                converted_path,
                "-y",
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            self.logger.info(f"Аудио конвертировано: {converted_path}")
            return converted_path

        except Exception as e:
            self.logger.error(f"Ошибка конвертации аудио: {e}")
            raise

    def _get_audio_duration_seconds(self, audio_path: str) -> Optional[float]:
        """Получить длительность аудио в секундах (torchaudio + ffprobe fallback)."""
        # 1) torchaudio.info (быстро)
        try:
            info = torchaudio.info(audio_path)
            sr = getattr(info, "sample_rate", None)
            if sr:
                return float(info.num_frames) / float(sr)
        except Exception:
            pass

        # 2) ffprobe (надёжнее для некоторых контейнеров/битых заголовков)
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]
            p = subprocess.run(cmd, check=True, capture_output=True, text=True)
            value = (p.stdout or "").strip()
            return float(value) if value else None
        except Exception:
            return None

    def _cap_segments(
        self, segments: List[Dict], max_len_s: float = 25.0
    ) -> List[Dict]:
        """Режем слишком длинные VAD-сегменты на куски, чтобы Whisper был стабильнее."""
        out = []
        idx = 0
        for seg in segments:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
            file_path = seg.get("file")
            if not file_path or end <= start:
                continue
            dur = end - start
            if dur <= max_len_s:
                out.append({**seg, "index": idx})
                idx += 1
                continue

            # Если сегмент длинный — пока просто добавляем без разбиения
            # (физическая нарезка wav требует дополнительной логики)
            out.append({**seg, "index": idx})
            idx += 1
        return out

    # ==================== ТРАНСКРИПЦИЯ ====================

    async def transcribe_audio(
        self, audio_path: str, skip_unload: bool = False
    ) -> List[Dict]:
        """
        Транскрипция аудио в текст с использованием Whisper
        Возвращает список сегментов с временными метками

        Args:
            audio_path: путь к аудиофайлу
            skip_unload: если True, не выгружает модель (для параллельной обработки)
        """
        self.logger.info(f"Начало транскрипции: {audio_path}")
        self.gpu_manager.take_snapshot("before_transcription")

        # Загружаем Whisper (без обновления прогресса - эта функция перенесена в параллельную обработку)
        success = await self.model_manager.load_whisper(skip_unload=skip_unload)
        if not success:
            raise Exception("Не удалось загрузить Whisper модель")

        try:
            result = self.model_manager.whisper_transcribe(audio_path)
            segments = []

            if isinstance(result.get("chunks"), list):
                for chunk in result["chunks"]:
                    segments.append(
                        {
                            "start": chunk["timestamp"][0],
                            "end": chunk["timestamp"][1],
                            "text": chunk["text"].strip(),
                        }
                    )
            else:
                # Fallback если нет chunks
                dur = self._get_audio_duration_seconds(audio_path) or 0.0
                segments.append(
                    {
                        "start": 0.0,
                        "end": float(dur) if dur > 0 else 0.0,
                        "text": (result.get("text", "") or "").strip(),
                    }
                )

            self.logger.info(f"Транскрипция завершена: {len(segments)} сегментов")
            self.gpu_manager.take_snapshot("after_transcription")

            return segments

        except Exception as e:
            self.logger.error(f"Ошибка транскрипции: {e}", exc_info=True)
            raise
        finally:
            # Выгружаем только если это не параллельная обработка
            if not skip_unload:
                await self.model_manager.unload_current_model()

    # ==================== ДИАРИЗАЦИЯ ====================

    async def diarize_audio(
        self, audio_path: str, skip_unload: bool = False
    ) -> List[Dict]:
        """
        Диаризация аудио - определение кто и когда говорил
        Использует PyAnnote для разделения по спикерам

        Args:
            audio_path: путь к аудиофайлу
            skip_unload: если True, не выгружает модель (для параллельной обработки)
        """
        self.logger.info(f"Начало диаризации: {audio_path}")
        self.gpu_manager.take_snapshot("before_diarization")

        # Загружаем PyAnnote (без обновления прогресса - эта функция перенесена в параллельную обработку)
        success = await self.model_manager.load_diarization(skip_unload=skip_unload)
        if not success:
            raise Exception("Не удалось загрузить PyAnnote модель")

        try:
            # Загружаем аудио в память
            waveform, sample_rate = torchaudio.load(audio_path)

            # Ресемплинг если нужно
            if sample_rate != 16000:
                self.logger.info(f"Ресемплинг: {sample_rate} Hz -> 16000 Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
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

            # PyAnnote может отключать TF32 ради воспроизводимости — возвращаем настройку проекта
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Запуск диаризации с оптимизированными параметрами
            # min_duration_off и min_duration_on уменьшают чувствительность,
            # что ускоряет обработку с минимальной потерей качества
            inputs = {"waveform": waveform, "sample_rate": sample_rate}
            output = self.model_manager.diarization_pipeline(
                inputs,
                min_duration_off=0.5,  # минимальная пауза между репликами (было 0.0)
                min_duration_on=0.5,  # минимальная длительность реплики (было 0.0)
            )

            # Извлечение результатов
            diarization = output.speaker_diarization
            result = []

            for turn, speaker in diarization:
                result.append(
                    {"start": turn.start, "end": turn.end, "speaker": speaker}
                )

            self.logger.info(f"Диаризация завершена: {len(result)} сегментов")
            self.gpu_manager.take_snapshot("after_diarization_run")

            # Очистка
            del waveform
            del inputs

            return result

        except Exception as e:
            self.logger.error(f"Ошибка диаризации: {e}", exc_info=True)
            raise
        finally:
            # Выгружаем только если это не параллельная обработка
            if not skip_unload:
                await self.model_manager.unload_current_model()

    # ==================== ОБЪЕДИНЕНИЕ ДАННЫХ ====================

    def align_transcription_and_diarization(
        self, transcription_segments: List[Dict], diarization_segments: List[Dict]
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
            if not isinstance(trans_start, (int, float)) or not isinstance(
                trans_end, (int, float)
            ):
                continue
            if float(trans_end) <= float(trans_start):
                continue

            # Находим пересечения со спикерами
            speaker_overlaps = {}

            for diar_seg in diarization_segments:
                overlap_start = max(trans_start, diar_seg["start"])
                overlap_end = min(trans_end, diar_seg["end"])
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > 0:
                    speaker = diar_seg["speaker"]
                    speaker_overlaps[speaker] = (
                        speaker_overlaps.get(speaker, 0) + overlap_duration
                    )

            # Определяем доминирующего спикера
            if speaker_overlaps:
                dominant_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
            else:
                dominant_speaker = "UNKNOWN"

            aligned.append(
                {
                    "start": trans_start,
                    "end": trans_end,
                    "speaker": dominant_speaker,
                    "text": trans_seg["text"],
                }
            )

        speakers_count = len(set(seg["speaker"] for seg in aligned))
        self.logger.info(
            f"Объединение завершено: {len(aligned)} сегментов, {speakers_count} спикеров"
        )

        return aligned

    # ==================== СУММАРИЗАЦИЯ ====================

    def _trim_to_word_boundary(self, text: str) -> str:
        text = text.rstrip()
        if not text:
            return text

        if re.search(r"[0-9A-Za-zА-Яа-яЁё]$", text):
            m = re.search(r"\s+\S+$", text)
            if m:
                text = text[: m.start()].rstrip()

        return text

    def _postprocess_summary(
        self,
        text: str,
        *,
        summary_type: str,
        stop_marker: str,
        hit_token_limit: bool,
    ) -> str:
        if not text:
            return ""

        if stop_marker in text:
            text = text.split(stop_marker, 1)[0]

        text = re.sub(r"(?mi)^\s*getStatusCode\(\)\s*$", "", text)
        text = re.sub(r"(?is)\bWrite a short summary of the meeting.*", "", text)

        if summary_type == "protocol":
            text = re.sub(
                r"(?s)\A\s*Дата проведения:\s*(?:\r?\n)\s*Присутствуют:\s*(?:\r?\n)\s*Повестка дня:\s*(?:\r?\n)\s*Рассмотрены вопросы и решения:\s*(?:\r?\n)\s*Итоги:\s*(?:\r?\n)+",
                "",
                text,
                count=1,
            )

        text = text.replace("**", "")
        text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
        text = re.sub(r"(?m)^\s*-{3,}\s*$", "", text)

        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if hit_token_limit:
            text = self._trim_to_word_boundary(text)

        return text.strip()

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
                    chunks.append(sentence[i : i + max_chars])
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
        max_tokens: int = 800,
    ) -> str:
        """Суммаризация одного чанка текста.

        Текущее исправление проблем суммаризации:
        - Возвращаем корректную остановку генерации (eos_token_id), чтобы модель не
          генерировала принудительно до max_new_tokens (это вызывало мусор и обрезание).
        - Добавляем строгий stop-marker + stopping_criteria, чтобы гарантированно
          остановиться в конце документа и не дописывать лишнее (например getStatusCode()).
        - Постобработка: убираем stop-marker/мусор и, если достигнут лимит токенов,
          обрезаем до границы слова.
        """

        stop_marker = (
            "<<<END_OF_PROTOCOL>>>"
            if summary_type == "protocol"
            else "<<<END_OF_SUMMARY>>>"
        )

        if summary_type == "standard":
            system_message = f"""Ты — секретарь, который делает краткое содержание видео на русском языке.

Сформируй краткое содержание СТРОГО по структуре ниже. Выводи ТОЛЬКО заполненную структуру (без комментариев/пояснений).

Структура:
Общая тематика видео: <...>
Основные темы:
— <...>
Краткое содержание:
<...>
Итоги:
— <...>

Требования к содержанию:
1) Восстанови главную линию обсуждения
2) Игнорируй "воду": слова-паразиты, междометия, эмоциональные вставки, повторы, оффтоп.
3) Если фраза не добавляет смысла, не используй её как опору для вывода.
4) Не додумывай и не исправляй факты: только то, что явно сказано в тексте.

Жёсткие требования:
- Только русский язык.
- Не добавляй приветствия, вводные фразы, пояснения.
- В конце выведи отдельной строкой: {stop_marker}"""
        else:
            system_message = f"""Ты — секретарь, оформляющий протокол совещания/конференции на русском языке.

Составь протокол СТРОГО по шаблону ниже. Выводи ТОЛЬКО заполненный шаблон (без пустого шаблона, без дублей, без комментариев).

Шаблон (порядок обязателен):
Дата проведения: <...>
Присутствуют: <...>
Повестка дня:
1) <...>
2) <...>
Рассмотрены вопросы и решения:
— <вопрос/обсуждение> — <решение/действие>
Итоги:
— <...>

Требования к содержанию:
1) Восстанови главную линию обсуждения
2) Игнорируй "воду": слова-паразиты, междометия, эмоциональные вставки, повторы, оффтоп.
3) Если фраза не добавляет смысла, не используй её как опору для вывода.
4) Не додумывай и не исправляй факты: только то, что явно сказано в тексте.
5) Если не была упомянута дата проведения, не придумывай свою

Жёсткие требования:
- Только русский язык.
- Никаких дополнительных разделов/абзацев вне шаблона.
- В конце выведи отдельной строкой: {stop_marker}"""

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Текст для анализа (не цитируй его в ответе):\n{text}",
            },
        ]

        prompt = self.model_manager.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.model_manager.qwen_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=24000,
        ).to(self.model_manager.qwen_model.device)

        input_length = len(inputs.input_ids[0])
        self.logger.info(
            f"Суммаризация чанка: input_tokens={input_length}, max_new_tokens={max_tokens}"
        )

        stop_token_ids = self.model_manager.qwen_tokenizer.encode(
            stop_marker, add_special_tokens=False
        )
        stop_token_ids_nl = self.model_manager.qwen_tokenizer.encode(
            "\n" + stop_marker, add_special_tokens=False
        )
        stop_criteria_items = [
            StopOnTokenSequence(stop_token_ids, start_length=input_length)
        ]
        if stop_token_ids_nl and stop_token_ids_nl != stop_token_ids:
            stop_criteria_items.append(
                StopOnTokenSequence(stop_token_ids_nl, start_length=input_length)
            )
        stopping_criteria = StoppingCriteriaList(stop_criteria_items)

        eos_token_id = self.model_manager.qwen_tokenizer.eos_token_id
        pad_token_id = self.model_manager.qwen_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        with torch.no_grad():
            outputs = self.model_manager.qwen_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                stopping_criteria=stopping_criteria,
            )

        generated_ids = outputs[0][input_length:].tolist()
        hit_token_limit = len(generated_ids) >= max_tokens
        self.logger.info(
            f"Суммаризация чанка: сгенерировано {len(generated_ids)} токенов (лимит {max_tokens}, достигнут={hit_token_limit})"
        )

        raw_text = self.model_manager.qwen_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        summary = self._postprocess_summary(
            raw_text,
            summary_type=summary_type,
            stop_marker=stop_marker,
            hit_token_limit=hit_token_limit,
        )

        return summary

    async def merge_summaries(
        self,
        summaries: List[str],
        summary_type: str = "standard",
    ) -> str:
        """Объединение нескольких суммаризаций в один итоговый документ.

        Ключевые моменты:
        - Генерация должна останавливаться корректно (EOS/stop-marker), иначе модель
          начинает дописывать «мусор» (в т.ч. getStatusCode()).
        - Используем тот же stop-marker, что и для summarize_chunk, и затем
          постобрабатываем результат.
        """

        stop_marker = (
            "<<<END_OF_PROTOCOL>>>"
            if summary_type == "protocol"
            else "<<<END_OF_SUMMARY>>>"
        )

        if summary_type == "standard":
            system_message = f"""Ты — секретарь. Объедини несколько кратких содержаний в ОДНО итоговое краткое содержание.

Структура:
Общая тематика видео: <...>
Основные темы:
— <...>
Краткое содержание:
<...>
Итоги:
— <...>

Требования к содержанию:
1) Восстанови главную линию обсуждения
2) Игнорируй "воду": слова-паразиты, междометия, эмоциональные вставки, повторы, оффтоп.
3) Если фраза не добавляет смысла, не используй её как опору для вывода.
4) Не додумывай и не исправляй факты: только то, что явно сказано в тексте.

Жёсткие требования:
- Только русский язык.
- Не добавляй приветствия, вводные фразы, пояснения.
- В конце выведи отдельной строкой: {stop_marker}"""
        else:
            system_message = f"""Ты — секретарь. Объедини несколько протоколов частей в ОДИН единый протокол.

Выводи результат СТРОГО по шаблону (порядок обязателен):
Дата проведения: <...>
Присутствуют: <...>
Повестка дня:
1) <...>
2) <...>
Рассмотрены вопросы и решения:
— <вопрос/обсуждение> — <решение/действие>
Итоги:
— <...>

Требования к содержанию:
1) Восстанови главную линию обсуждения
2) Игнорируй "воду": слова-паразиты, междометия, эмоциональные вставки, повторы, оффтоп.
3) Если фраза не добавляет смысла, не используй её как опору для вывода.
4) Не додумывай и не исправляй факты: только то, что явно сказано в тексте.
5) Если не была упомянута дата проведения, не придумывай свою

Жёсткие требования:
- Только русский язык.
- Никаких дополнительных разделов/абзацев вне шаблона.
- В конце выведи отдельной строкой: {stop_marker}"""

        combined_text = "\n\n".join(summaries)

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Части для объединения (не цитируй дословно, а объедини и избавься от дублей):\n{combined_text}",
            },
        ]

        prompt = self.model_manager.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        max_new_tokens = AppConfig.SUMMARY_MAX_NEW_TOKENS.get("final", 7000)
        inputs = self.model_manager.qwen_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=32000,
        ).to(self.model_manager.qwen_model.device)

        input_length = len(inputs.input_ids[0])
        self.logger.info(
            f"Объединение суммаризаций: input_tokens={input_length}, max_new_tokens={max_new_tokens}"
        )

        stop_token_ids = self.model_manager.qwen_tokenizer.encode(
            stop_marker, add_special_tokens=False
        )
        stop_token_ids_nl = self.model_manager.qwen_tokenizer.encode(
            "\n" + stop_marker, add_special_tokens=False
        )
        stop_criteria_items = [
            StopOnTokenSequence(stop_token_ids, start_length=input_length)
        ]
        if stop_token_ids_nl and stop_token_ids_nl != stop_token_ids:
            stop_criteria_items.append(
                StopOnTokenSequence(stop_token_ids_nl, start_length=input_length)
            )
        stopping_criteria = StoppingCriteriaList(stop_criteria_items)

        eos_token_id = self.model_manager.qwen_tokenizer.eos_token_id
        pad_token_id = self.model_manager.qwen_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        with torch.no_grad():
            outputs = self.model_manager.qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                stopping_criteria=stopping_criteria,
            )

        generated_ids = outputs[0][input_length:].tolist()
        hit_token_limit = len(generated_ids) >= max_new_tokens
        self.logger.info(
            f"Объединение суммаризаций: сгенерировано {len(generated_ids)} токенов (лимит {max_new_tokens}, достигнут={hit_token_limit})"
        )

        raw_text = self.model_manager.qwen_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        final_summary = self._postprocess_summary(
            raw_text,
            summary_type=summary_type,
            stop_marker=stop_marker,
            hit_token_limit=hit_token_limit,
        )

        return final_summary

    async def summarize_text(self, text: str, summary_type: str = "standard") -> str:
        """
        Главный метод суммаризации
        Разбивает текст на части, суммаризует каждую и объединяет
        """
        if not text.strip():
            raise ValueError("Текст пуст")

        self.logger.info(f"Начало суммаризации ({summary_type}): {len(text)} символов")
        self.gpu_manager.take_snapshot("before_summarization")

        # Этап 6: Загружаем Qwen (обновление прогресса уже делается в process_transcription_and_diarization_parallel)
        self.logger.info("Загрузка Qwen модели...")
        success = await self.model_manager.load_qwen()
        if not success:
            raise Exception("Не удалось загрузить Qwen модель")

        # Этап 7: Делаем краткое содержание (обновление прогресса уже делается в process_transcription_and_diarization_parallel)
        self.logger.info("✅ Qwen загружен успешно")

        try:
            # Разбиваем текст на части
            max_chars = AppConfig.SUMMARIZATION_MAX_CHARS
            chunks = self.split_text_into_chunks(text, max_chars=max_chars)
            total_chunks = len(chunks)

            self.logger.info(f"Текст разбит на {total_chunks} частей")

            # Суммаризуем каждую часть
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                self.logger.info(f"Суммаризация части {i+1}/{total_chunks}")

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
                final_summary = await self.merge_summaries(
                    chunk_summaries, summary_type
                )

            self.logger.info(f"Суммаризация завершена: {len(final_summary)} символов")
            self.gpu_manager.take_snapshot("after_summarization")

            return final_summary

        except Exception as e:
            self.logger.error(f"Ошибка суммаризации: {e}", exc_info=True)
            raise
        finally:
            await self.model_manager.unload_current_model()

    # ==================== ФОРМАТИРОВАНИЕ И СОХРАНЕНИЕ ====================
    @staticmethod
    def _format_ts(seconds: float) -> str:
        """Формат [MM:SS] или [HH:MM:SS]."""
        try:
            s = int(round(float(seconds)))
        except Exception:
            s = 0

        if s < 0:
            s = 0

        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60

        if h > 0:
            return f"{h:02d}:{m:02d}:{sec:02d}"
        return f"{m:02d}:{sec:02d}"


    def format_transcription(self, segments: List[Dict]) -> str:
        """
        Форматирование транскрипции с таймкодами и блоками по спикерам:
        СПИКЕР: SPEAKER_07
        ------------------------------
        [00:00] Текст...
        """
        if not segments:
            return ""

        segments = sorted(segments, key=lambda s: float(s.get("start", 0.0) or 0.0))

        formatted = ""
        current_speaker = None

        for segment in segments:
            speaker = segment.get("speaker") or "UNKNOWN"
            text = (segment.get("text") or "").strip()
            start = segment.get("start")

            if not text:
                continue

            if speaker != current_speaker:
                if formatted:
                    formatted += "\n"
                formatted += f"СПИКЕР: {speaker}\n"
                formatted += "------------------------------\n"
                current_speaker = speaker

            ts = self._format_ts(start if isinstance(start, (int, float)) else 0.0)
            formatted += f"[{ts}] {text}\n"

        return formatted.rstrip() + "\n"


    def add_metadata(
        self,
        content: str,
        task_id: str,
        processing_time: float,
        media_type: str,
        stats: Dict,
    ) -> str:
        """
        Добавление метаданных к результату
        """
        tomsk_time = self._get_tomsk_time()

        metadata = f"""# Отчет обработки

**ID задачи**: {task_id}
**Дата**: {tomsk_time.strftime('%Y-%m-%d %H:%M:%S')} (Томск, UTC+7)
**Время обработки**: {processing_time:.2f} мин
**Длительность аудио**: {((stats.get('audio_duration_seconds') or 0) / 60):.1f} мин
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

    # ==================== ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА ====================

    async def process_transcription_and_diarization_parallel(
        self,
        audio_path: str,
        *,
        audio_duration_seconds: Optional[float] = None,
    ) -> tuple:
        """Параллельная обработка транскрипции и диаризации.

        Загружает обе модели и запускает их одновременно для экономии времени.

        Returns:
            кортеж (transcription_segments, diarization_segments)
        """
        duration_hint = ""
        if (
            isinstance(audio_duration_seconds, (int, float))
            and audio_duration_seconds > 0
        ):
            duration_hint = (
                f" (длительность аудио: {audio_duration_seconds / 60:.1f} мин)"
            )

        # Этап 2: Загружаем AI модели
        self._update_progress(
            10,
            "loading_ai_models",
            f"Загрузка моделей Whisper и PyAnnote...{duration_hint}",
        )
        self.logger.info("Запуск параллельной обработки транскрипции и диаризации")
        self.logger.info("Загрузка модели Whisper...")

        try:
            # Загружаем Whisper
            whisper_result = await self.model_manager.load_whisper(skip_unload=True)
            if not whisper_result:
                raise Exception("Не удалось загрузить Whisper модель")

            # Небольшая задержка, чтобы UI успел обновиться
            await asyncio.sleep(0.1)

            # Обновляем прогресс после загрузки Whisper
            self._update_progress(
                12, "loading_ai_models", "Whisper загружен, загрузка PyAnnote..."
            )
            self.logger.info("Загрузка модели PyAnnote...")

            # Загружаем PyAnnote
            diarization_result = await self.model_manager.load_diarization(
                skip_unload=True
            )
            if not diarization_result:
                raise Exception("Не удалось загрузить PyAnnote модель")

            # Небольшая задержка, чтобы UI успел обновиться
            await asyncio.sleep(0.1)

            # Этап 3: Делаем расшифровку
            self._update_progress(
                20, "transcribing", "Распознавание речи и определение спикеров..."
            )
            self.logger.info("Обе модели загружены, запуск параллельной обработки...")

            # Запускаем оба процесса параллельно
            transcription_segments, diarization_segments = await asyncio.gather(
                self._transcribe_audio_parallel(audio_path),
                self._diarize_audio_parallel(audio_path),
                return_exceptions=False,
            )

            # Важно: обновляем этапы завершения только ПОСЛЕ того как завершились ОБА процесса,
            # иначе пользователь может увидеть "Диаризация завершена" пока транскрипция еще идет (или наоборот).
            self._update_progress(
                50, "transcription_completed", "Транскрипция завершена"
            )
            # Небольшая задержка, чтобы UI успел обновиться
            await asyncio.sleep(0.1)
            self._update_progress(55, "diarization_completed", "Диаризация завершена")

            self.logger.info("Параллельная обработка завершена")
            return transcription_segments, diarization_segments

        except Exception as e:
            self.logger.error(f"Ошибка параллельной обработки: {e}", exc_info=True)
            raise
        finally:
            # Выгружаем обе модели после завершения
            await self.model_manager.unload_whisper_and_diarization()

    async def _transcribe_audio_parallel(self, audio_path: str) -> List[Dict]:
        """Вспомогательный метод для параллельной транскрипции"""
        import time

        try:
            start_time = time.time()
            self.logger.info("🔄 [WHISPER] Начало транскрипции аудио...")

            # Используем run_in_executor для запуска синхронного кода в отдельном потоке
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self.model_manager.whisper_transcribe, audio_path
            )

            segments = []
            if isinstance(result.get("chunks"), list):
                for chunk in result["chunks"]:
                    segments.append(
                        {
                            "start": chunk["timestamp"][0],
                            "end": chunk["timestamp"][1],
                            "text": chunk["text"].strip(),
                        }
                    )
            else:
                segments.append(
                    {"start": 0.0, "end": None, "text": result.get("text", "")}
                )

            elapsed = time.time() - start_time
            self.logger.info(
                f"✅ [WHISPER] Транскрипция завершена: {len(segments)} сегментов за {elapsed:.2f}с"
            )
            self.gpu_manager.take_snapshot("after_transcription")

            return segments

        except Exception as e:
            self.logger.error(f"❌ [WHISPER] Ошибка транскрипции: {e}", exc_info=True)
            raise

    async def _diarize_audio_parallel(self, audio_path: str) -> List[Dict]:
        """Вспомогательный метод для параллельной диаризации"""
        import time

        try:
            start_time = time.time()
            self.logger.info("🔄 [PYANNOTE] Начало диаризации аудио...")

            # Функция для выполнения диаризации (будет запущена в отдельном потоке)
            def run_diarization():
                # Загружаем аудио в память
                waveform, sample_rate = torchaudio.load(audio_path)

                # Ресемплинг если нужно
                if sample_rate != 16000:
                    self.logger.info(
                        f"[PYANNOTE] Ресемплинг: {sample_rate} Hz -> 16000 Hz"
                    )
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=16000
                    )
                    waveform = resampler(waveform)
                    sample_rate = 16000

                # Конвертация в моно если стерео
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Перенос на GPU если доступно
                if torch.cuda.is_available():
                    waveform = waveform.to("cuda")
                    # Убедимся что waveform в float32 для совместимости с PyAnnote
                    waveform = waveform.to(torch.float32)

                self.logger.info(
                    f"[PYANNOTE] Аудио подготовлено: shape={waveform.shape}, dtype={waveform.dtype}"
                )

                # PyAnnote может отключать TF32 ради воспроизводимости — возвращаем настройку проекта
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

                # Запуск диаризации с оптимизированными параметрами
                inputs = {"waveform": waveform, "sample_rate": sample_rate}
                output = self.model_manager.diarization_pipeline(
                    inputs,
                    min_duration_off=0.5,  # минимальная пауза между репликами
                    min_duration_on=0.5,  # минимальная длительность реплики
                )

                # Извлечение результатов
                diarization = output.speaker_diarization
                result = []

                for turn, speaker in diarization:
                    result.append(
                        {"start": turn.start, "end": turn.end, "speaker": speaker}
                    )

                # Очистка
                del waveform
                del inputs

                return result

            # Используем run_in_executor для запуска в отдельном потоке
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, run_diarization)

            elapsed = time.time() - start_time
            self.logger.info(
                f"✅ [PYANNOTE] Диаризация завершена: {len(result)} сегментов за {elapsed:.2f}с"
            )
            self.gpu_manager.take_snapshot("after_diarization_run")

            return result

        except Exception as e:
            self.logger.error(f"❌ [PYANNOTE] Ошибка диаризации: {e}", exc_info=True)
            raise

    # ==================== ГЛАВНЫЙ ПАЙПЛАЙН ====================

    async def process_media(
        self,
        file_path: str,
        task_id: str,
        media_type: str = "video",
        summary_type: str = "standard",
    ) -> Dict:
        """
        Главный метод обработки медиа файла с улучшениями аудио

        Выполняет полный цикл:
        1. Подготовка аудио (извлечение/конвертация)
        2. Speech Enhancement (DeepFilterNet)
        3. Нормализация громкости (EBU R128)
        4. VAD сегментация (Silero)
        5. Транскрипция VAD-сегментов (Whisper)
        6. Пост-фильтрация результатов
        7. Диаризация (PyAnnote)
        8. Объединение транскрипции и диаризации
        9. Суммаризация текста (Qwen)
        10. Форматирование и сохранение
        """
        # Этап 1: Начинаем обрабатывать задачу
        self._update_progress(0, "task_started", "Начало обработки задачи")
        self.logger.info("=" * 60)
        self.logger.info(f"НАЧАЛО ОБРАБОТКИ ЗАДАЧИ: {task_id}")
        self.logger.info(f"Тип: {media_type}, Суммаризация: {summary_type}")
        self.logger.info("=" * 60)

        start_time = datetime.now()
        audio_path = None
        enhanced_path = None
        normalized_path = None
        vad_dir = None
        created_temp_files: List[str] = []

        try:
            # Небольшая задержка, чтобы UI успел подключиться и увидеть начальный статус
            await asyncio.sleep(0.1)

            # Очистка перед началом
            await self.gpu_manager.cleanup("standard")
            self.gpu_manager.take_snapshot("initial")

            # === ЭТАП 1: ПОДГОТОВКА АУДИО ===
            self._update_progress(2, "preparing_audio", "Подготовка аудиофайла...")
            self.logger.info("Подготовка аудиофайла...")
            if media_type == "audio":
                audio_path = self.process_audio_file(file_path, task_id)
            else:
                audio_path = self.extract_audio(file_path, task_id)
            created_temp_files.append(audio_path)

            audio_duration_seconds = self._get_audio_duration_seconds(audio_path)
            if audio_duration_seconds:
                self.logger.info(
                    f"Длительность аудио: {audio_duration_seconds:.2f} сек"
                )
            else:
                self.logger.warning(
                    "Не удалось определить длительность аудио (torchaudio+ffprobe)"
                )

            # === ЭТАП 2: SPEECH ENHANCEMENT (DeepFilterNet) ===
            self._update_progress(
                5, "enhancing_audio", "Улучшение качества аудио (шумоподавление)..."
            )
            self.logger.info("Применение DeepFilterNet для шумоподавления...")
            enhanced_path = audio_path.replace(".wav", "_enhanced.wav")

            loop = asyncio.get_running_loop()
            enhancement_success = await loop.run_in_executor(
                None, enhance_audio, audio_path, enhanced_path
            )

            if not enhancement_success:
                self.logger.warning("⚠️ Enhancement failed, используем исходное аудио")
                enhanced_path = audio_path
            else:
                created_temp_files.append(enhanced_path)

            # === ЭТАП 3: НОРМАЛИЗАЦИЯ ГРОМКОСТИ ===
            self._update_progress(8, "normalizing_audio", "Выравнивание громкости...")
            self.logger.info("Нормализация громкости (EBU R128)...")
            normalized_path = enhanced_path.replace(".wav", "_normalized.wav")
            if enhanced_path == audio_path:
                normalized_path = audio_path.replace(".wav", "_normalized.wav")

            normalization_success = await loop.run_in_executor(
                None, normalize_loudness, enhanced_path, normalized_path
            )

            if not normalization_success:
                self.logger.warning("⚠️ Нормализация failed, используем предыдущий файл")
                normalized_path = enhanced_path
            else:
                created_temp_files.append(normalized_path)

            # Финальный аудио путь для обработки
            final_audio_path = normalized_path

            # === ЭТАП 4: VAD СЕГМЕНТАЦИЯ ===
            self._update_progress(
                10, "vad_segmentation", "Выделение речевых фрагментов (VAD)..."
            )
            self.logger.info("Применение Silero VAD для сегментации...")
            vad_dir = os.path.join(self.upload_dir, f"{task_id}_vad")

            vad_segments = await loop.run_in_executor(
                None, split_by_vad, final_audio_path, vad_dir
            )

            # Если VAD не нашёл сегментов — используем весь файл
            use_vad_segments = len(vad_segments) > 0
            if not use_vad_segments:
                self.logger.warning(
                    "⚠️ VAD не обнаружил речевых сегментов, используем весь аудио файл"
                )
                end_ts = (
                    float(audio_duration_seconds) if audio_duration_seconds else None
                )
                vad_segments = [
                    {
                        "index": 0,
                        "start": 0.0,
                        "end": end_ts,
                        "duration": float(end_ts) if end_ts else None,
                        "file": final_audio_path,
                    }
                ]
            else:
                self.logger.info(f"VAD обнаружил {len(vad_segments)} речевых сегментов")
                # Дополнительно капаем длину сегментов
                vad_segments = self._cap_segments(vad_segments, max_len_s=25.0)

            # === ЭТАП 5: ЗАГРУЗКА МОДЕЛЕЙ ===
            duration_hint = ""
            if (
                isinstance(audio_duration_seconds, (int, float))
                and audio_duration_seconds > 0
            ):
                duration_hint = (
                    f" (длительность аудио: {audio_duration_seconds / 60:.1f} мин)"
                )

            self._update_progress(
                12,
                "loading_ai_models",
                f"Загрузка моделей Whisper и PyAnnote...{duration_hint}",
            )
            self.logger.info("Загрузка AI моделей...")

            # Загружаем Whisper
            whisper_result = await self.model_manager.load_whisper(skip_unload=True)
            if not whisper_result:
                raise Exception("Не удалось загрузить Whisper модель")

            await asyncio.sleep(0.1)
            self._update_progress(
                14, "loading_ai_models", "Whisper загружен, загрузка PyAnnote..."
            )

            # Загружаем PyAnnote
            diarization_result = await self.model_manager.load_diarization(
                skip_unload=True
            )
            if not diarization_result:
                raise Exception("Не удалось загрузить PyAnnote модель")

            await asyncio.sleep(0.1)

            # === ЭТАП 6: ТРАНСКРИПЦИЯ VAD-СЕГМЕНТОВ ===
            self._update_progress(
                20,
                "transcribing",
                f"Распознавание речи ({len(vad_segments)} сегментов)...",
            )
            self.logger.info(f"Транскрипция {len(vad_segments)} VAD-сегментов...")

            all_transcription_chunks = []
            total_segments = len(vad_segments)

            for i, seg in enumerate(vad_segments):
                # Обновляем прогресс каждые 10 сегментов
                if i % 10 == 0:
                    progress = 20 + int((i / total_segments) * 20)  # 20-40%
                    self._update_progress(
                        progress,
                        "transcribing",
                        f"Распознавание речи: {i+1}/{total_segments}",
                    )

                # Транскрибируем сегмент
                result = await loop.run_in_executor(
                    None, self.model_manager.whisper_transcribe, seg["file"]
                )

                # Добавляем offset к таймкодам
                if isinstance(result.get("chunks"), list):
                    for chunk in result["chunks"]:
                        ts = chunk.get("timestamp")
                        if not isinstance(ts, (list, tuple)) or len(ts) != 2:
                            continue
                        if ts[0] is None or ts[1] is None:
                            continue
                        if float(ts[1]) <= float(ts[0]):
                            continue

                        start = float(ts[0]) + float(seg.get("start", 0.0) or 0.0)
                        end = float(ts[1]) + float(seg.get("start", 0.0) or 0.0)
                        text = (chunk.get("text") or "").strip()
                        if not text:
                            continue

                        all_transcription_chunks.append(
                            {
                                "start": start,
                                "end": end,
                                "text": text,
                            }
                        )
                elif result.get("text"):
                    # fallback без чанков: только если есть внятный интервал
                    seg_start = float(seg.get("start", 0.0) or 0.0)
                    seg_end = seg.get("end")
                    if isinstance(seg_end, (int, float)) and seg_end > seg_start:
                        all_transcription_chunks.append(
                            {
                                "start": seg_start,
                                "end": float(seg_end),
                                "text": (result.get("text") or "").strip(),
                            }
                        )

            self.logger.info(
                f"✅ Транскрипция завершена: {len(all_transcription_chunks)} сегментов"
            )
            self.gpu_manager.take_snapshot("after_transcription")

            # === ЭТАП 7: ПОСТ-ФИЛЬТРАЦИЯ ===
            self._update_progress(42, "filtering_results", "Фильтрация результатов...")
            self.logger.info("Пост-фильтрация результатов транскрипции...")

            transcription_segments = filter_transcription_results(
                all_transcription_chunks
            )
            self.logger.info(
                f"После фильтрации: {len(transcription_segments)} сегментов"
            )

            # === ЭТАП 8: ДИАРИЗАЦИЯ ===
            self._update_progress(45, "diarization_running", "Определение спикеров...")
            self.logger.info("Запуск диаризации на полном аудио...")

            diarization_segments = await self._diarize_audio_parallel(final_audio_path)

            self.logger.info(
                f"✅ Диаризация завершена: {len(diarization_segments)} сегментов"
            )
            self._update_progress(55, "diarization_completed", "Диаризация завершена")

            # Выгружаем Whisper и PyAnnote
            await self.model_manager.unload_whisper_and_diarization()

            # === ЭТАП 9: ОБЪЕДИНЕНИЕ ===
            aligned_segments = self.align_transcription_and_diarization(
                transcription_segments, diarization_segments
            )

            # Извлекаем полный текст и статистику
            full_text = " ".join([seg["text"] for seg in aligned_segments])
            speakers_count = len(set(seg["speaker"] for seg in aligned_segments))

            # === ЭТАП 10: СУММАРИЗАЦИЯ ===
            await asyncio.sleep(0.1)

            self._update_progress(
                60, "loading_qwen", "Загрузка модели суммаризации (Qwen)..."
            )
            summary = await self.summarize_text(full_text, summary_type)

            await asyncio.sleep(0.1)

            self._update_progress(70, "summarizing", "Создание краткого содержания...")
            await asyncio.sleep(0.1)

            # === ФОРМАТИРОВАНИЕ И СОХРАНЕНИЕ ===
            formatted_transcription = self.format_transcription(aligned_segments)
            formatted_transcription = remove_profanity_transcription(
                formatted_transcription
            )
            processing_time = (datetime.now() - start_time).total_seconds() / 60

            # Получаем статистику GPU
            gpu_stats = self.gpu_manager.get_memory_stats()

            stats = {
                "segments_count": len(aligned_segments),
                "speakers_count": speakers_count,
                "audio_duration_seconds": audio_duration_seconds,
                "peak_gpu_usage": gpu_stats.get("peak_usage_gb", 0),
                "peak_gpu_percent": gpu_stats.get("peak_usage_percent", 0),
                "peak_stage": gpu_stats.get("peak_stage", "unknown"),
                "vad_segments_count": len(vad_segments) if use_vad_segments else 0,
            }

            # Добавляем метаданные
            formatted_transcription = self.add_metadata(
                formatted_transcription, task_id, processing_time, media_type, stats
            )

            # Сохраняем файлы
            self.save_result(
                os.path.join(self.results_dir, f"{task_id}_transcription.md"),
                formatted_transcription,
            )
            self.save_result(
                os.path.join(self.results_dir, f"{task_id}_summary.md"),
                summary,
            )

            # Выводим сводку по памяти
            self.gpu_manager.log_memory_summary()

            # Все готово
            self._update_progress(100, "task_completed", "Обработка успешно завершена")

            self.logger.info("=" * 60)
            self.logger.info(f"ЗАДАЧА ЗАВЕРШЕНА: {task_id}")
            self.logger.info(f"Время обработки: {processing_time:.2f} мин")
            self.logger.info(
                f"Сегментов: {len(aligned_segments)}, Спикеров: {speakers_count}"
            )
            self.logger.info("=" * 60)

            return {
                "task_id": task_id,
                "summary": summary,
                "transcription_length": len(formatted_transcription),
                "segments_count": len(aligned_segments),
                "speakers_count": speakers_count,
                "processing_time_minutes": round(processing_time, 2),
                "audio_duration_seconds": (
                    round(audio_duration_seconds, 2)
                    if isinstance(audio_duration_seconds, (int, float))
                    else None
                ),
                "media_type": media_type,
                "gpu_peak_usage_gb": gpu_stats.get("peak_usage_gb", 0),
                "vad_segments_count": len(vad_segments) if use_vad_segments else 0,
            }

        except Exception as e:
            self.logger.error("=" * 60)
            self.logger.error(f"КРИТИЧЕСКАЯ ОШИБКА В ЗАДАЧЕ: {task_id}")
            self.logger.error(f"Ошибка: {e}", exc_info=True)
            self.logger.error("=" * 60)
            raise

        finally:
            # Очистка моделей
            await self.model_manager.unload_all_models()

            # Удаляем только файлы, которые мы точно создали как временные
            safe_remove = []
            for p in created_temp_files:
                if not p:
                    continue
                try:
                    if (
                        os.path.exists(p)
                        and Path(p).parent.resolve() == Path(self.upload_dir).resolve()
                    ):
                        safe_remove.append(p)
                except Exception:
                    pass
            cleanup_temp_files(*safe_remove)

            # Удаление VAD директории
            if vad_dir:
                cleanup_vad_directory(vad_dir)

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
            "device": (
                self.model_manager.device
                if hasattr(self, "model_manager")
                else "unknown"
            ),
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

