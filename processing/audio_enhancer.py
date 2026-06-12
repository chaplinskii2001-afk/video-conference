"""
Модуль обработки аудио для video-conference-processor

Функции:
- enhance_audio / enhanceaudio: (DeepFilterNet ОТКЛЮЧЕН) только приведение к 16kHz mono WAV через ffmpeg
- normalize_loudness / normalizeloudness: EBU R128 нормализация + компрессия
- split_by_vad / splitbyvad: VAD-сегментация через Silero
- filter_transcription_results / filtertranscriptionresults: пост-фильтрация результатов Whisper
- cleanup_temp_files / cleanuptempfiles: очистка временных файлов
- cleanup_vad_directory / cleanupvaddirectory: очистка директории VAD
"""

import os
import subprocess
import logging
import glob
import re
from typing import List, Dict, Optional
from collections import Counter

import torch
import torchaudio

logger = logging.getLogger(__name__)

# Единые дефолты (должны совпадать с Dockerfile / compose)
os.environ.setdefault("XDG_CACHE_HOME", "/app/models/xdg_cache")
os.environ.setdefault("TORCH_HOME", "/app/models/torch_home")

_SILERO_VAD = None
_SILERO_UTILS = None


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================


def _ffprobe_duration_seconds(path: str) -> Optional[float]:
    """Получить длительность аудио через ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        p = subprocess.run(cmd, check=True, capture_output=True, text=True)
        value = (p.stdout or "").strip()
        if not value:
            return None
        return float(value)
    except Exception:
        return None


# ==================== SPEECH ENHANCEMENT (DF OFF) ====================


def _resample_to_16k_mono(input_path: str, output_path: str) -> bool:
    """Приведение к 16kHz mono WAV для Whisper."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error(f"❌ ffmpeg resample stderr: {stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ ffmpeg resample error: {e}", exc_info=True)
        return False


def enhance_audio(input_path: str, output_path: str) -> bool:
    """
    DeepFilterNet отключён. Функция делает только ресемпл 16kHz mono.
    DISABLE_DF оставлен для совместимости и явного лога.
    """
    if os.getenv("DISABLE_DF", "0") == "1":
        logger.info("DeepFilterNet отключен (DISABLE_DF=1) — ресемпл 16k mono")
    else:
        logger.info("DeepFilterNet отключен (hard-off) — ресемпл 16k mono")

    return _resample_to_16k_mono(input_path, output_path)


# Совместимость со старыми вызовами/ветками:
def enhance_audio_chunked(
    input_path: str, output_path: str, *, chunk_seconds: int = 60
) -> bool:
    # DF отключен, chunked не нужен — просто приводим к 16k mono.
    return enhance_audio(input_path, output_path)


def enhance_audio_python(input_path: str, output_path: str) -> bool:
    # DF отключен — делаем ресемпл.
    return enhance_audio(input_path, output_path)


def enhance_audio_deepfilternet(input_path: str, output_path: str) -> bool:
    # DF отключен — делаем ресемпл.
    return enhance_audio(input_path, output_path)


# Алиас под то, как импортируется в video_processor.py
enhanceaudio = enhance_audio


# ==================== НОРМАЛИЗАЦИЯ ГРОМКОСТИ ====================


def normalize_loudness(input_path: str, output_path: str) -> bool:
    """
    Нормализация громкости + компрессия динамики.

    - loudnorm: EBU R128 нормализация
    - acompressor: мягкая компрессия (поднимает тихие голоса)
    - highpass: убираем низкочастотный гул
    - lowpass: убираем высокочастотный свист
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-af",
            "loudnorm=I=-16:LRA=11:TP=-1.5,"
            "acompressor=threshold=-18dB:ratio=3:attack=5:release=50,"
            "highpass=f=80,"
            "lowpass=f=8000",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"✅ Нормализация громкости: {input_path} → {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error(f"❌ Нормализация ошибка: {stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Нормализация ошибка: {e}", exc_info=True)
        return False


normalizeloudness = normalize_loudness


# ==================== VAD (SILERO) ====================


def merge_close_segments(segments: List[Dict], max_gap_ms: int = 400) -> List[Dict]:
    """Склеивает сегменты с паузой меньше max_gap_ms."""
    if not segments:
        return []
    max_gap_samples = int(max_gap_ms * 16)  # 16 samples/ms для 16kHz
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"]
        if gap <= max_gap_samples:
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged


def split_by_vad(audio_path: str, output_dir: str) -> List[Dict]:
    """
    Применяет Silero VAD для выделения речевых сегментов.
    Возвращает [{'index': i, 'start': sec, 'end': sec, 'duration': sec, 'file': path}, ...]
    """
    try:
        global _SILERO_VAD, _SILERO_UTILS
        if _SILERO_VAD is None or _SILERO_UTILS is None:
            _SILERO_VAD, _SILERO_UTILS = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = (
            _SILERO_UTILS
        )

        wav = read_audio(audio_path, sampling_rate=16000)

        speech_timestamps = get_speech_timestamps(
            wav,
            _SILERO_VAD,
            threshold=0.3,
            min_speech_duration_ms=250,
            min_silence_duration_ms=400,
            window_size_samples=512,
            speech_pad_ms=100,
        )

        if not speech_timestamps:
            logger.warning("⚠️ VAD: речь не обнаружена")
            return []

        merged_segments = merge_close_segments(speech_timestamps, max_gap_ms=400)

        os.makedirs(output_dir, exist_ok=True)
        segments: List[Dict] = []
        for i, seg in enumerate(merged_segments):
            start_sample = seg["start"]
            end_sample = seg["end"]

            chunk_audio = wav[start_sample:end_sample]
            chunk_path = os.path.join(output_dir, f"vad_chunk_{i:04d}.wav")
            torchaudio.save(chunk_path, chunk_audio.unsqueeze(0), 16000)

            segments.append(
                {
                    "index": i,
                    "start": start_sample / 16000,
                    "end": end_sample / 16000,
                    "duration": (end_sample - start_sample) / 16000,
                    "file": chunk_path,
                }
            )

        logger.info(f"✅ VAD: {len(segments)} речевых сегментов выделено")
        return segments

    except Exception as e:
        logger.error(
            "Silero VAD не удалось загрузить. "
            "Если включён офлайн-режим — прогрей VAD на этапе сборки Dockerfile командой "
            "python -c \"import torch; torch.hub.load('snakers4/silero-vad','silero_vad', trust_repo=True)\". "
            f"Ошибка: {e}",
            exc_info=True,
        )
        return []


splitbyvad = split_by_vad


# ==================== ПОСТ-ФИЛЬТРАЦИЯ ====================


def is_similar(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Jaccard-подобное сравнение."""
    norm1 = re.sub(r"[^0-9A-Za-zА-Яа-яЁё\s]+", " ", (text1 or "").lower()).strip()
    norm2 = re.sub(r"[^0-9A-Za-zА-Яа-яЁё\s]+", " ", (text2 or "").lower()).strip()
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    if not words1 or not words2:
        return False
    return (len(words1 & words2) / len(words1 | words2)) >= threshold


def filter_transcription_results(segments: List[Dict]) -> List[Dict]:
    """
    Пост-фильтрация результатов Whisper.
    """
    if not segments:
        return []

    garbage_patterns = {
        "♪",
        "♪♪",
        "♪♪♪",
        "[MUSIC]",
        "(music)",
        "music",
        "Thank you.",
        "Thanks for watching.",
        "thank you",
        "thanks for watching",
    }

    filtered: List[Dict] = []
    prev_text = ""

    def _is_mostly_garbage(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        alnum = sum(ch.isalnum() for ch in t)
        if alnum == 0:
            return True
        if alnum / max(len(t), 1) < 0.35:
            return True
        return False

    def _chars_per_second(seg: Dict) -> Optional[float]:
        try:
            start = seg.get("start")
            end = seg.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                return None
            dur = float(end) - float(start)
            if dur <= 0:
                return None
            text = (seg.get("text") or "").strip()
            meaningful = re.sub(r"\s+", "", re.sub(r"[^\wА-Яа-яЁё0-9]+", "", text))
            return len(meaningful) / dur
        except Exception:
            return None

    def _word_repetition_ratio(text: str) -> float:
        t = re.sub(r"[^0-9A-Za-zА-Яа-яЁё\s]+", " ", (text or "").lower()).strip()
        words = [w for w in t.split() if len(w) > 1]
        if len(words) < 6:
            return 0.0
        c = Counter(words)
        most_common = c.most_common(1)[0][1]
        return most_common / len(words)

    for seg in segments:
        text = (seg.get("text") or "").strip()

        if not text or len(text) < 2:
            continue
        if text in garbage_patterns:
            continue
        if _is_mostly_garbage(text):
            continue

        start = seg.get("start")
        end = seg.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            dur = float(end) - float(start)
            if dur > 0 and dur < 0.25 and len(text) >= 15:
                continue
            cps = _chars_per_second(seg)
            if cps is not None and cps > 35:
                continue

        rep = _word_repetition_ratio(text)
        if rep >= 0.45:
            continue

        if text == prev_text:
            continue
        if prev_text and is_similar(text, prev_text, threshold=0.8):
            continue

        filtered.append(seg)
        prev_text = text

    if len(filtered) != len(segments):
        logger.info(
            f"📋 Фильтрация транскрипции: {len(segments)} → {len(filtered)} сегментов"
        )

    return filtered


filtertranscriptionresults = filter_transcription_results


# ==================== ОЧИСТКА ВРЕМЕННЫХ ФАЙЛОВ ====================


def cleanup_temp_files(*file_paths: str) -> None:
    """Удаляет временные файлы."""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"🗑️ Удалён временный файл: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось удалить {file_path}: {e}")


cleanuptempfiles = cleanup_temp_files


def cleanup_vad_directory(vad_dir: str) -> None:
    """Удаляет директорию с VAD сегментами."""
    if vad_dir and os.path.exists(vad_dir):
        try:
            import shutil

            shutil.rmtree(vad_dir)
            logger.debug(f"🗑️ Удалена директория VAD: {vad_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось удалить директорию {vad_dir}: {e}")


cleanupvaddirectory = cleanup_vad_directory

