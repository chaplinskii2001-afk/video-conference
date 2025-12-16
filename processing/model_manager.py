"""
Менеджер AI моделей
Управляет загрузкой, выгрузкой и переключением между моделями
"""

import os
import torch
import logging
import warnings
from typing import Optional, Dict, Set, Any

from transformers import (
    pipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
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

    В проекте допускается параллельная загрузка Whisper + PyAnnote для экспериментов с VRAM.
    """

    def __init__(self, config: Dict, gpu_manager: GPUMemoryManager):
        self.config = config
        self.gpu_manager = gpu_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.hf_token = os.getenv("HF_TOKEN")

        self.whisper_model = None
        self.whisper_processor = None
        self.whisper_pipeline = None

        self.diarization_pipeline = None

        self.qwen_model = None
        self.qwen_tokenizer = None

        self.loaded_models: Set[str] = set()

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

    @property
    def current_loaded_model(self) -> Optional[str]:
        if not self.loaded_models:
            return None
        return "+".join(sorted(self.loaded_models))

    def _resolve_torch_dtype(self, value: Any, *, default: torch.dtype) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            if normalized in mapping:
                return mapping[normalized]

        return default

    async def _maybe_unload_others(self, keep: str, *, unload_others: bool) -> None:
        if not unload_others:
            return

        to_unload = set(self.loaded_models)
        to_unload.discard(keep)
        await self.unload_models(to_unload)

    # ==================== WHISPER ====================

    async def load_whisper(self, *, unload_others: bool = True) -> bool:
        """Загрузка модели Whisper для транскрипции"""

        if self.whisper_pipeline is not None and "whisper" in self.loaded_models:
            self.logger.info("Whisper уже загружен")
            return True

        await self._maybe_unload_others("whisper", unload_others=unload_others)

        self.logger.info("Загрузка Whisper модели...")
        self.gpu_manager.take_snapshot("before_whisper")

        try:
            whisper_path = self.config["model_config"]["whisper_path"]
            whisper_id = self.config["model_config"]["whisper_id"]

            quantization = self.gpu_config.get("whisper_quantization", "8bit")
            whisper_8bit_config = self.gpu_config.get("whisper_8bit_config", {})

            self.logger.info(f"Whisper квантование: {quantization}")

            self.whisper_processor = WhisperProcessor.from_pretrained(
                whisper_id,
                cache_dir=whisper_path,
            )

            if quantization == "8bit":
                try:
                    load_in_8bit = whisper_8bit_config["load_in_8bit"]
                    dtype_value = whisper_8bit_config["dtype"]
                except KeyError as e:
                    raise ValueError(
                        "Некорректная конфигурация квантования Whisper (whisper_8bit_config) в config.py"
                    ) from e

                dtype = self._resolve_torch_dtype(
                    dtype_value,
                    default=torch.float16 if self.device == "cuda:0" else torch.float32,
                )

                model = WhisperForConditionalGeneration.from_pretrained(
                    whisper_id,
                    cache_dir=whisper_path,
                    load_in_8bit=load_in_8bit,
                    device_map="auto",
                    dtype=dtype,
                )
            elif quantization == "float16":
                model = WhisperForConditionalGeneration.from_pretrained(
                    whisper_id,
                    cache_dir=whisper_path,
                    device_map="auto",
                    dtype=torch.float16,
                )
            else:  # float32
                model = WhisperForConditionalGeneration.from_pretrained(
                    whisper_id,
                    cache_dir=whisper_path,
                    device_map="auto",
                    dtype=torch.float32,
                )

            self.logger.info(
                "Whisper параметры: "
                f"chunk_length_s={self.whisper_chunk_length_s}, "
                f"stride_length_s={self.whisper_stride_length_s}, "
                f"batch_size={self.whisper_batch_size}, "
                f"max_audio_length_minutes={self.max_audio_length_minutes}"
            )

            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=self.whisper_processor.tokenizer,
                feature_extractor=self.whisper_processor.feature_extractor,
                chunk_length_s=self.whisper_chunk_length_s,
                stride_length_s=self.whisper_stride_length_s,
                return_timestamps=True,
            )

            self.loaded_models.add("whisper")
            self.gpu_manager.take_snapshot("after_whisper")
            self.logger.info("✅ Whisper загружен успешно")
            return True

        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки Whisper: {e}", exc_info=True)
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
        if self.whisper_pipeline is None or "whisper" not in self.loaded_models:
            raise RuntimeError("Whisper pipeline не загружен")

        self._enforce_max_audio_length(audio_path)

        try:
            return self.whisper_pipeline(audio_path, batch_size=self.whisper_batch_size)
        except TypeError:
            return self.whisper_pipeline(audio_path)

    # ==================== PYANNOTE (DIARIZATION) ====================

    async def load_diarization(self, *, unload_others: bool = True) -> bool:
        """Загрузка модели PyAnnote для диаризации спикеров"""

        if self.diarization_pipeline is not None and "diarization" in self.loaded_models:
            self.logger.info("PyAnnote уже загружен")
            return True

        await self._maybe_unload_others("diarization", unload_others=unload_others)

        self.logger.info("Загрузка PyAnnote модели...")
        self.gpu_manager.take_snapshot("before_diarization")

        try:
            diarization_id = self.config["model_config"]["diarization_id"]
            diarization_path = self.config["model_config"]["diarization_path"]

            self.diarization_pipeline = Pipeline.from_pretrained(
                diarization_id,
                token=self.hf_token,
                cache_dir=diarization_path,
            )

            if torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
                self.logger.info("PyAnnote перенесен на GPU")

                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            self.loaded_models.add("diarization")
            self.gpu_manager.take_snapshot("after_diarization")
            self.logger.info("✅ PyAnnote загружен успешно")
            return True

        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки PyAnnote: {e}", exc_info=True)
            self.logger.error("Проверьте HF_TOKEN и права доступа к модели")
            await self.gpu_manager.cleanup("deep")
            return False

    # ==================== QWEN (SUMMARIZATION) ====================

    async def load_qwen(self, *, unload_others: bool = True) -> bool:
        """Загрузка модели Qwen для суммаризации"""

        if self.qwen_model is not None and "qwen" in self.loaded_models:
            self.logger.info("Qwen уже загружен")
            return True

        await self._maybe_unload_others("qwen", unload_others=unload_others)

        self.logger.info("Загрузка Qwen модели...")
        self.gpu_manager.take_snapshot("before_qwen")

        try:
            qwen_path = self.config["model_config"]["qwen_path"]

            if not os.path.exists(qwen_path):
                raise FileNotFoundError(f"Qwen модель не найдена: {qwen_path}")

            quantization = self.gpu_config.get("qwen_quantization", "4bit")
            qwen_4bit_bnb_config = self.gpu_config.get("qwen_4bit_bnb_config", {})
            qwen_8bit_config = self.gpu_config.get("qwen_8bit_config", {})

            self.logger.info(f"Qwen квантование: {quantization}")

            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                qwen_path,
                trust_remote_code=True,
                local_files_only=True,
            )

            if quantization == "4bit":
                try:
                    load_in_4bit = qwen_4bit_bnb_config["load_in_4bit"]
                    use_double_quant = qwen_4bit_bnb_config["use_double_quant"]
                    quant_type = qwen_4bit_bnb_config["quant_type"]
                    compute_dtype_value = qwen_4bit_bnb_config["compute_dtype"]
                except KeyError as e:
                    raise ValueError(
                        "Некорректная конфигурация квантования Qwen (qwen_4bit_bnb_config) в config.py"
                    ) from e

                compute_dtype = self._resolve_torch_dtype(
                    compute_dtype_value,
                    default=torch.float16,
                )
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_use_double_quant=use_double_quant,
                    bnb_4bit_quant_type=quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                )

                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    qwen_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    dtype=compute_dtype,
                )

            elif quantization == "8bit":
                try:
                    load_in_8bit = qwen_8bit_config["load_in_8bit"]
                    dtype_value = qwen_8bit_config["dtype"]
                except KeyError as e:
                    raise ValueError(
                        "Некорректная конфигурация квантования Qwen (qwen_8bit_config) в config.py"
                    ) from e

                model_dtype = self._resolve_torch_dtype(
                    dtype_value,
                    default=torch.float16,
                )

                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    qwen_path,
                    load_in_8bit=load_in_8bit,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    dtype=model_dtype,
                )

            else:  # float16 или float32
                dtype = torch.float16 if quantization == "float16" else torch.float32
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    qwen_path,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    dtype=dtype,
                )

            self.loaded_models.add("qwen")
            self.gpu_manager.take_snapshot("after_qwen")
            self.logger.info("✅ Qwen загружен успешно")
            return True

        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки Qwen: {e}", exc_info=True)
            await self.gpu_manager.cleanup("deep")
            return False

    # ==================== УПРАВЛЕНИЕ ====================

    async def unload_models(self, model_names: Set[str], cleanup_mode: str = "standard"):
        """Выгрузка набора моделей."""

        did_unload = False

        for model_name in sorted(model_names):
            if model_name not in self.loaded_models:
                continue

            self.logger.info(f"Выгрузка модели: {model_name}")

            try:
                if model_name == "whisper":
                    if self.whisper_pipeline is not None:
                        del self.whisper_pipeline
                    if self.whisper_processor is not None:
                        del self.whisper_processor
                    self.whisper_pipeline = None
                    self.whisper_processor = None

                elif model_name == "diarization":
                    if self.diarization_pipeline is not None:
                        del self.diarization_pipeline
                    self.diarization_pipeline = None

                elif model_name == "qwen":
                    if self.qwen_model is not None:
                        del self.qwen_model
                    if self.qwen_tokenizer is not None:
                        del self.qwen_tokenizer
                    self.qwen_model = None
                    self.qwen_tokenizer = None

                self.loaded_models.discard(model_name)
                did_unload = True

            except Exception as e:
                self.logger.warning(f"Ошибка при выгрузке модели {model_name}: {e}")

        if did_unload:
            await self.gpu_manager.cleanup(cleanup_mode)

    async def unload_current_model(self):
        """Выгрузка текущей загруженной модели (совместимость со старым API)."""

        if not self.loaded_models:
            return

        if len(self.loaded_models) == 1:
            await self.unload_models(set(self.loaded_models))
        else:
            await self.unload_all_models()

    async def unload_all_models(self):
        """Выгрузка всех моделей"""

        self.logger.info("Выгрузка всех моделей")
        await self.unload_models(set(self.loaded_models), cleanup_mode="deep")

    def get_loaded_models_info(self) -> Dict:
        """Информация о загруженных моделях"""

        return {
            "whisper_loaded": self.whisper_pipeline is not None,
            "diarization_loaded": self.diarization_pipeline is not None,
            "qwen_loaded": self.qwen_model is not None,
            "loaded_models": sorted(self.loaded_models),
            "current_model": self.current_loaded_model,
            "device": self.device,
            "whisper_quantization": self.gpu_config.get("whisper_quantization", "unknown"),
            "qwen_quantization": self.gpu_config.get("qwen_quantization", "unknown"),
            "chunk_length_s": self.whisper_chunk_length_s,
            "stride_length_s": self.whisper_stride_length_s,
            "batch_size": self.whisper_batch_size,
            "max_audio_length_minutes": self.max_audio_length_minutes,
        }
