"""
Управление GPU памятью с детальным мониторингом
Отслеживает использование памяти на каждом этапе обработки
"""
import gc
import asyncio
import logging
import torch
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml недоступен, детальный мониторинг GPU отключен")


@dataclass
class GPUMemorySnapshot:
    """Снимок состояния памяти GPU"""
    timestamp: datetime
    stage: str
    total_gb: float
    used_gb: float
    free_gb: float
    usage_percent: float
    allocated_torch_gb: float
    reserved_torch_gb: float


class GPUMemoryManager:
    """
    Менеджер памяти GPU
    - Отслеживает использование памяти на каждом этапе
    - Логирует изменения
    - Выполняет очистку при необходимости
    """
    
    def __init__(self, log_memory_changes: bool = True):
        self.log_memory_changes = log_memory_changes
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.memory_snapshots = []
        self.nvml_initialized = False
        self.nvml_handle = None
        
        # Инициализация NVML если доступно
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.logger.info("NVML инициализирован успешно")
            except Exception as e:
                self.logger.warning(f"Не удалось инициализировать NVML: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Получить текущее состояние памяти GPU
        Возвращает словарь с информацией о памяти
        """
        if not torch.cuda.is_available():
            return {
                "total_gb": 0.0,
                "used_gb": 0.0,
                "free_gb": 0.0,
                "usage_percent": 0.0,
                "allocated_torch_gb": 0.0,
                "reserved_torch_gb": 0.0
            }
        
        try:
            # Информация от PyTorch
            allocated_bytes = torch.cuda.memory_allocated(0)
            reserved_bytes = torch.cuda.memory_reserved(0)
            allocated_gb = allocated_bytes / (1024 ** 3)
            reserved_gb = reserved_bytes / (1024 ** 3)
            
            # Информация от NVML (если доступно)
            if self.nvml_initialized and self.nvml_handle:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                total_gb = info.total / (1024 ** 3)
                used_gb = info.used / (1024 ** 3)
                free_gb = info.free / (1024 ** 3)
                usage_percent = (info.used / info.total) * 100
            else:
                # Резервные значения от PyTorch
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                used_gb = reserved_gb
                free_gb = total_gb - used_gb
                usage_percent = (used_gb / total_gb) * 100 if total_gb > 0 else 0
            
            return {
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "free_gb": round(free_gb, 2),
                "usage_percent": round(usage_percent, 1),
                "allocated_torch_gb": round(allocated_gb, 2),
                "reserved_torch_gb": round(reserved_gb, 2)
            }
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о памяти GPU: {e}")
            return {
                "total_gb": 0.0,
                "used_gb": 0.0,
                "free_gb": 0.0,
                "usage_percent": 0.0,
                "allocated_torch_gb": 0.0,
                "reserved_torch_gb": 0.0
            }
    
    def take_snapshot(self, stage: str) -> GPUMemorySnapshot:
        """
        Создать снимок памяти на текущем этапе
        Полезно для отслеживания утечек памяти
        """
        memory_info = self.get_memory_info()
        snapshot = GPUMemorySnapshot(
            timestamp=datetime.now(),
            stage=stage,
            total_gb=memory_info["total_gb"],
            used_gb=memory_info["used_gb"],
            free_gb=memory_info["free_gb"],
            usage_percent=memory_info["usage_percent"],
            allocated_torch_gb=memory_info["allocated_torch_gb"],
            reserved_torch_gb=memory_info["reserved_torch_gb"]
        )
        
        self.memory_snapshots.append(snapshot)
        
        if self.log_memory_changes:
            self.logger.info(
                f"📊 GPU [{stage}]: "
                f"Использовано {snapshot.used_gb:.2f}/{snapshot.total_gb:.2f} GB "
                f"({snapshot.usage_percent:.1f}%), "
                f"PyTorch: {snapshot.allocated_torch_gb:.2f} GB выделено"
            )
        
        return snapshot
    
    async def cleanup(self, level: str = "standard"):
        """
        Очистка памяти GPU
        
        Уровни:
        - light: быстрая очистка кэша
        - standard: стандартная очистка с синхронизацией
        - deep: глубокая очистка с паузой
        """
        if not torch.cuda.is_available():
            return
        
        memory_before = self.get_memory_info()
        
        if level == "light":
            # Быстрая очистка
            gc.collect()
            torch.cuda.empty_cache()
            await asyncio.sleep(0.1)
        
        elif level == "standard":
            # Стандартная очистка
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            await asyncio.sleep(0.5)
        
        elif level == "deep":
            # Глубокая очистка
            gc.collect()
            gc.collect()  # Двойной вызов для надежности
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            await asyncio.sleep(1.0)
        
        memory_after = self.get_memory_info()
        freed_gb = memory_before["used_gb"] - memory_after["used_gb"]
        
        if self.log_memory_changes and abs(freed_gb) > 0.01:
            self.logger.info(
                f"🧹 Очистка GPU ({level}): "
                f"освобождено {freed_gb:.2f} GB, "
                f"свободно {memory_after['free_gb']:.2f} GB"
            )
    
    def log_memory_summary(self):
        """Вывести сводку по использованию памяти за все этапы"""
        if not self.memory_snapshots:
            return
        
        self.logger.info("=" * 60)
        self.logger.info("📈 СВОДКА ИСПОЛЬЗОВАНИЯ ПАМЯТИ GPU")
        self.logger.info("=" * 60)
        
        for snapshot in self.memory_snapshots:
            self.logger.info(
                f"{snapshot.stage:20s}: "
                f"{snapshot.used_gb:.2f} GB использовано "
                f"({snapshot.usage_percent:.1f}%)"
            )
        
        # Пиковое использование
        peak_snapshot = max(self.memory_snapshots, key=lambda s: s.used_gb)
        self.logger.info("=" * 60)
        self.logger.info(
            f"Пиковое использование: {peak_snapshot.used_gb:.2f} GB "
            f"на этапе '{peak_snapshot.stage}'"
        )
        self.logger.info("=" * 60)
    
    def get_memory_stats(self) -> Dict:
        """Получить статистику использования памяти"""
        if not self.memory_snapshots:
            return {}
        
        used_values = [s.used_gb for s in self.memory_snapshots]
        percent_values = [s.usage_percent for s in self.memory_snapshots]
        
        peak_snapshot = max(self.memory_snapshots, key=lambda s: s.used_gb)
        
        return {
            "peak_usage_gb": round(max(used_values), 2),
            "peak_usage_percent": round(max(percent_values), 1),
            "peak_stage": peak_snapshot.stage,
            "average_usage_gb": round(sum(used_values) / len(used_values), 2),
            "snapshots_count": len(self.memory_snapshots)
        }
    
    def clear_snapshots(self):
        """Очистить сохраненные снимки"""
        self.memory_snapshots.clear()
    
    def __del__(self):
        """Очистка при удалении объекта"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
