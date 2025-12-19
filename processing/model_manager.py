"""
Менеджер AI моделей
Управляет загрузкой, выгрузкой и переключением между моделями
"""
import os
import torch
import logging
import warnings
from typing import Optional, Dict, List, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Подавляем предупреждения о generation flags в transformers
warnings.filterwarnings("ignore", message=r".*generation flags are not valid.*")

# Подавляем предупреждение PyAnnote о выключении TF32 (в проекте TF32 включен для ускорения)
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
)
warnings.filterwarnings(
    "ignore",
    module=r"pyannote\.audio\.utils\.reproducibility",
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
        self.whisper_backend = None

        self.diarization_pipeline = None

        self.qwen_model = None
        self.qwen_tokenizer = None

        # Вспомогательная модель VAD (используется для удаления тишины перед диаризацией)
        self._silero_vad_model = None
        self._silero_get_speech_timestamps = None

        self.current_loaded_model = None
        
        # Получаем конфигурацию GPU
        self.gpu_config = config.get("gpu_config", {})
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        try:
            self.whisper_batch_size = int(self.gpu_config["batch_size"])
            self.max_audio_length_minutes = self.gpu_config["max_audio_length_minutes"]
            self.whisper_chunk_length_s = self.gpu_config["chunk_length_s"]
            self.whisper_stride_length_s = self.gpu_config["stride_length_s"]
        except KeyError as e:
            raise ValueError(
                "Некорректная gpu_config: ожидаются ключи batch_size, max_audio_length_minutes, "
                "chunk_length_s, stride_length_s"
            ) from e

        self.logger.info(f"ModelManager инициализирован для устройства: {self.device}")
        self.logger.info(
            "Параметры профиля: "
            f"batch_size={self.whisper_batch_size}, "
            f"max_audio_length_minutes={self.max_audio_length_minutes}, "
            f"chunk_length_s={self.whisper_chunk_length_s}, "
            f"stride_length_s={self.whisper_stride_length_s}"
        )
    
    # ==================== WHISPER (faster-whisper / CTranslate2) ====================

    async def load_whisper(self, skip_unload: bool = False) -> bool:
        """Загрузка модели Whisper для транскрипции через faster-whisper (CTranslate2).

        faster-whisper существенно быстрее классического transformers pipeline и поддерживает
        int8 compute types, что уменьшает потребление VRAM.

        Args:
            skip_unload: если True, не выгружает текущую модель (для параллельной загрузки)
        """
        if self.whisper_model is not None:
            self.logger.info("Whisper уже загружен (faster-whisper)")
            return True

        if not skip_unload:
            await self.unload_current_model()

        self.logger.info("Загрузка Whisper модели (faster-whisper)...")
        self.gpu_manager.take_snapshot("before_whisper")

        try:
            from faster_whisper import WhisperModel

            whisper_path = self.config["model_config"]["whisper_path"]
            os.makedirs(whisper_path, exist_ok=True)
            whisper_id = self.gpu_config.get("whisper_model_id") or self.config["model_config"]["whisper_id"]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = self.gpu_config.get("whisper_compute_type")
            if not compute_type:
                quantization = self.gpu_config.get("whisper_quantization", "int8")
                if quantization in ("8bit", "int8"):
                    compute_type = "int8_float16" if device == "cuda" else "int8"
                elif quantization == "float16":
                    compute_type = "float16"
                else:
                    compute_type = "float32"

            num_workers = int(self.gpu_config.get("whisper_num_workers", 1))
            cpu_threads = int(self.gpu_config.get("whisper_cpu_threads", 0))

            self.logger.info(
                "faster-whisper параметры: "
                f"model_id={whisper_id}, device={device}, compute_type={compute_type}, "
                f"chunk_length={self.whisper_chunk_length_s}, batch_size={self.whisper_batch_size}, "
                f"num_workers={num_workers}, cpu_threads={cpu_threads}"
            )

            self.whisper_model = WhisperModel(
                whisper_id,
                device=device,
                device_index=0,
                compute_type=compute_type,
                download_root=whisper_path,
                num_workers=num_workers,
                cpu_threads=cpu_threads,
            )
            self.whisper_backend = "faster-whisper"

            if not skip_unload:
                self.current_loaded_model = "whisper"

            self.gpu_manager.take_snapshot("after_whisper")
            self.logger.info("✅ Whisper загружен успешно (faster-whisper)")
            return True

        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки Whisper (faster-whisper): {e}", exc_info=True)
            await self.gpu_manager.cleanup("deep")
            return False

    def _get_audio_duration_minutes(self, audio_path: str) -> Optional[float]:
        try:
            import torchaudio

            info = torchaudio.info(audio_path)
            sample_rate = getattr(info, "sample_rate", None)
            if not sample_rate:
                return None

            return (info.num_frames / sample_rate) / 60
        except Exception as e:
            self.logger.warning(f"Не удалось определить длительность аудио: {e}")
            return None

    def _enforce_max_audio_length(self, audio_path: str) -> None:
        if not self.max_audio_length_minutes:
            return

        duration_minutes = self._get_audio_duration_minutes(audio_path)
        if duration_minutes is None:
            return

        if duration_minutes > self.max_audio_length_minutes:
            raise ValueError(
                "Длина аудио превышает лимит профиля: "
                f"{duration_minutes:.1f} мин > {self.max_audio_length_minutes} мин"
            )

    def whisper_transcribe(self, audio_path: str) -> Dict:
        if self.whisper_model is None:
            raise RuntimeError("Whisper model не загружен")

        self._enforce_max_audio_length(audio_path)

        beam_size = int(self.gpu_config.get("whisper_beam_size", 5))
        language = self.gpu_config.get("whisper_language")
        condition_on_previous_text = bool(self.gpu_config.get("whisper_condition_on_previous_text", True))

        kwargs = {
            "beam_size": beam_size,
            "language": language,
            "condition_on_previous_text": condition_on_previous_text,
            "chunk_length": int(self.whisper_chunk_length_s),
            "vad_filter": False,
            "batch_size": int(self.whisper_batch_size),
        }

        try:
            import inspect

            allowed = set(inspect.signature(self.whisper_model.transcribe).parameters.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        except Exception:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

        segments, info = self.whisper_model.transcribe(audio_path, **kwargs)

        chunks = []
        texts = []
        for segment in segments:
            chunks.append(
                {
                    "timestamp": [float(segment.start), float(segment.end)],
                    "text": segment.text.strip(),
                }
            )
            texts.append(segment.text)

        return {
            "text": "".join(texts).strip(),
            "chunks": chunks,
            "language": getattr(info, "language", None),
        }

    # ==================== PYANNOTE (DIARIZATION) ====================
    
    async def load_diarization(self, skip_unload: bool = False) -> bool:
        """
        Загрузка модели PyAnnote для диаризации спикеров
        
        Args:
            skip_unload: если True, не выгружает текущую модель (для параллельной загрузки)
        """
        if self.diarization_pipeline is not None:
            self.logger.info("PyAnnote уже загружен")
            return True
        
        # Выгружаем другие модели только если это не параллельная загрузка
        if not skip_unload:
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

                # PyAnnote может отключать TF32 ради воспроизводимости — возвращаем настройку проекта
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Оптимизация параметров диаризации для скорости
            # Увеличиваем batch_size для embedding (ускоряет обработку)
            diarization_batch_size = self.gpu_config.get("diarization_batch_size", 32)
            if hasattr(self.diarization_pipeline, '_embedding'):
                try:
                    self.diarization_pipeline._embedding.batch_size = diarization_batch_size
                    self.logger.info(f"PyAnnote embedding batch_size установлен: {diarization_batch_size}")
                except Exception as e:
                    self.logger.warning(f"Не удалось установить batch_size для embedding: {e}")
            
            # Обновляем current_loaded_model только если это не параллельная загрузка
            if not skip_unload:
                self.current_loaded_model = "diarization"
            self.gpu_manager.take_snapshot("after_diarization")
            
            self.logger.info("✅ PyAnnote загружен успешно")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки PyAnnote: {e}", exc_info=True)
            self.logger.error("Проверьте HF_TOKEN и права доступа к модели")
            await self.gpu_manager.cleanup("deep")
            return False

    # ==================== SILERO VAD ====================

    def _ensure_silero_vad_loaded(self) -> None:
        if self._silero_vad_model is not None and self._silero_get_speech_timestamps is not None:
            return

        try:
            from silero_vad import load_silero_vad, get_speech_timestamps

            self._silero_vad_model = load_silero_vad()
            self._silero_vad_model.eval()
            self._silero_get_speech_timestamps = get_speech_timestamps
            self.logger.info("✅ Silero VAD загружен (silero-vad)")
            return
        except Exception as e:
            self.logger.info(f"silero-vad пакет недоступен, пробуем torch.hub: {e}")

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (get_speech_timestamps, _, _, _, _) = utils
        self._silero_vad_model = model
        self._silero_vad_model.eval()

        def _get_ts(audio: torch.Tensor, model: torch.nn.Module, sampling_rate: int, **kwargs: Any):
            ts = get_speech_timestamps(audio, model, sampling_rate=sampling_rate, **kwargs)
            if kwargs.get("return_seconds"):
                return ts
            return ts

        self._silero_get_speech_timestamps = _get_ts
        self.logger.info("✅ Silero VAD загружен (torch.hub)")

    def get_speech_timestamps(
        self,
        audio_mono_16khz: torch.Tensor,
        *,
        sample_rate: int,
        threshold: float = 0.5,
        speech_pad_ms: int = 250,
    ) -> List[Dict[str, int]]:
        """Возвращает интервалы речи (в samples) для Silero VAD.

        Важно: вход ожидается как 1D tensor (float32) на CPU.
        """

        self._ensure_silero_vad_loaded()

        if audio_mono_16khz.ndim != 1:
            raise ValueError("Silero VAD ожидает 1D tensor")

        audio_cpu = audio_mono_16khz.detach().to(torch.float32).cpu()

        # silero-vad использует sampling_rate=16000 в большинстве кейсов
        if sample_rate != 16000:
            raise ValueError(f"Silero VAD ожидает sample_rate=16000, получен: {sample_rate}")

        timestamps = self._silero_get_speech_timestamps(
            audio_cpu,
            self._silero_vad_model,
            sampling_rate=sample_rate,
            threshold=threshold,
            speech_pad_ms=speech_pad_ms,
        )

        return [
            {"start": int(item["start"]), "end": int(item["end"])}
            for item in (timestamps or [])
            if isinstance(item, dict) and "start" in item and "end" in item
        ]

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
            
            enable_flash_attn2 = bool(self.gpu_config.get("qwen_flash_attention_2", False))
            enable_compile = bool(self.gpu_config.get("qwen_torch_compile", False))

            attn_implementation = None
            if enable_flash_attn2:
                if not torch.cuda.is_available():
                    self.logger.info("Flash Attention 2 пропущен: CUDA недоступен")
                elif quantization != "float16":
                    self.logger.info(
                        "Flash Attention 2 пропущен: требуется qwen_quantization=float16 (текущее: %s)",
                        quantization,
                    )
                else:
                    try:
                        import flash_attn  # noqa: F401

                        attn_implementation = "flash_attention_2"
                        self.logger.info("✅ Flash Attention 2 будет использован для Qwen")
                    except Exception as e:
                        self.logger.warning(f"Flash Attention 2 недоступен (pip install flash-attn): {e}")

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

                model_kwargs = {
                    "trust_remote_code": True,
                    "local_files_only": True,
                    "torch_dtype": dtype,
                }
                if attn_implementation:
                    model_kwargs["attn_implementation"] = attn_implementation

                # torch.compile корректно работает, когда модель не загружена через device_map (accelerate hooks).
                if torch.cuda.is_available() and quantization == "float16" and (enable_compile or attn_implementation):
                    self.qwen_model = AutoModelForCausalLM.from_pretrained(qwen_path, **model_kwargs)
                    self.qwen_model = self.qwen_model.to(torch.device("cuda"))
                else:
                    device_map = "auto" if torch.cuda.is_available() else None
                    self.qwen_model = AutoModelForCausalLM.from_pretrained(
                        qwen_path,
                        device_map=device_map,
                        **model_kwargs,
                    )

            if enable_compile:
                if not hasattr(torch, "compile"):
                    self.logger.info("torch.compile недоступен: требуется torch>=2.0")
                elif not torch.cuda.is_available():
                    self.logger.info("torch.compile пропущен: CUDA недоступен")
                elif quantization != "float16":
                    self.logger.info(
                        "torch.compile пропущен: требуется qwen_quantization=float16 (текущее: %s)",
                        quantization,
                    )
                elif getattr(self.qwen_model, "hf_device_map", None):
                    self.logger.info("torch.compile пропущен: модель загружена через device_map")
                else:
                    try:
                        mode = self.gpu_config.get("qwen_torch_compile_mode", "reduce-overhead")
                        self.qwen_model = torch.compile(self.qwen_model, mode=mode)
                        self.logger.info(f"✅ torch.compile включен для Qwen (mode={mode})")
                    except Exception as e:
                        self.logger.warning(f"torch.compile не удалось включить: {e}")

            self.qwen_model.eval()
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
                del self.whisper_model
                self.whisper_model = None
                self.whisper_backend = None

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
    
    async def unload_whisper_and_diarization(self):
        """Выгрузка Whisper и PyAnnote моделей (используется после параллельной обработки)"""
        self.logger.info("Выгрузка моделей Whisper и PyAnnote")
        
        try:
            if self.whisper_model:
                del self.whisper_model
            if self.diarization_pipeline:
                del self.diarization_pipeline

            self.whisper_model = None
            self.whisper_backend = None
            self.diarization_pipeline = None
            
            if self.current_loaded_model in ("whisper", "diarization"):
                self.current_loaded_model = None
            
            await self.gpu_manager.cleanup("standard")
            
        except Exception as e:
            self.logger.warning(f"Ошибка при выгрузке моделей: {e}")
    
    async def unload_all_models(self):
        """Выгрузка всех моделей"""
        self.logger.info("Выгрузка всех моделей")
        
        if self.whisper_model:
            del self.whisper_model
        if self.diarization_pipeline:
            del self.diarization_pipeline
        if self.qwen_model:
            del self.qwen_model
            del self.qwen_tokenizer

        self.whisper_model = None
        self.whisper_backend = None
        self.diarization_pipeline = None
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.current_loaded_model = None
        
        await self.gpu_manager.cleanup("deep")
    
    def get_loaded_models_info(self) -> Dict:
        """Информация о загруженных моделях"""
        return {
            "whisper_loaded": self.whisper_model is not None,
            "whisper_backend": self.whisper_backend,
            "whisper_model_id": self.gpu_config.get("whisper_model_id") or self.config.get("model_config", {}).get("whisper_id"),
            "whisper_compute_type": self.gpu_config.get("whisper_compute_type"),
            "diarization_loaded": self.diarization_pipeline is not None,
            "qwen_loaded": self.qwen_model is not None,
            "current_model": self.current_loaded_model,
            "device": self.device,
            "qwen_quantization": self.gpu_config.get("qwen_quantization", "unknown"),
            "qwen_flash_attention_2": self.gpu_config.get("qwen_flash_attention_2", False),
            "qwen_torch_compile": self.gpu_config.get("qwen_torch_compile", False),
            "chunk_length_s": self.whisper_chunk_length_s,
            "stride_length_s": self.whisper_stride_length_s,
            "batch_size": self.whisper_batch_size,
            "max_audio_length_minutes": self.max_audio_length_minutes,
        }
