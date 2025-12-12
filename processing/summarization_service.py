"""
Сервис для работы с моделью Qwen суммаризации.
Отвечает за создание кратких содержаний и протоколов.
"""
import os
import re
import asyncio
import logging
from typing import List, Dict, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import gc
from config import get_settings, get_model_config


logger = logging.getLogger(__name__)


class SummarizationService:
    """Сервис суммаризации текста с помощью Qwen"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = get_settings().QWEN_MODEL_PATH
        self.model_config = get_model_config()
    
    async def initialize(self) -> None:
        """Инициализация модели Qwen"""
        if self.model is not None:
            return
            
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель Qwen не найдена по пути: {self.model_path}")
        
        logger.info("Инициализация Qwen модели для суммаризации")
        
        await self._cleanup_memory()
        
        try:
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Конфигурация 4-битного квантования
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            # Загружаем модель
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float16,
            )
            
            logger.info("Qwen модель загружена успешно")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки Qwen модели: {e}")
            await self._cleanup_memory()
            raise
    
    async def summarize(
        self, 
        text: str, 
        summary_type: str = "standard",
        max_length: int = 4000
    ) -> str:
        """
        Создание суммаризации текста
        
        Args:
            text: Исходный текст
            summary_type: Тип суммаризации ("standard" или "protocol")
            max_length: Максимальная длина суммаризации
            
        Returns:
            Сгенерированная суммаризация
        """
        if not text.strip():
            raise ValueError("Текст для суммаризации пуст")
        
        await self.initialize()
        
        try:
            logger.info(f"Создание суммаризации типа '{summary_type}'. Длина текста: {len(text)}")
            
            # Выбираем тип суммаризации
            if summary_type == "protocol":
                summary = await self._summarize_protocol(text, max_length)
            else:
                summary = await self._summarize_standard(text, max_length)
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Ошибка суммаризации: {e}")
            raise
    
    async def _summarize_standard(self, text: str, max_length: int) -> str:
        """Стандартная суммаризация"""
        prompt = f"""Ты - ассистент для создания кратких содержаний. Создай краткое содержание следующего текста:

{text}

Краткое содержание:"""

        return await self._generate_summary(prompt, max_length)
    
    async def _summarize_protocol(self, text: str, max_length: int) -> str:
        """Суммаризация в формате протокола"""
        prompt = f"""Ты - секретарь, оформляющий протокол видео на русском языке. Используй следующий шаблон:

Шаблон:
Общая тематика видео:
(краткое описание основной темы)

Ключевые участники:
(список участников и их роли)

Основные решения:
• 
• 

Выводы и рекомендации:

Создай протокол для следующего текста:

{text}

Протокол:"""

        return await self._generate_summary(prompt, max_length)
    
    async def _generate_summary(self, prompt: str, max_length: int) -> str:
        """Генерация суммаризации с помощью модели"""
        # Токенизация входного текста
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=24000
        ).to(self.model.device)
        
        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.model_config.QWEN_MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=self.model_config.QWEN_REPETITION_PENALTY,
            )
        
        # Извлекаем только новые токены
        generated_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        summary = self.tokenizer.decode(
            generated_ids, 
            skip_special_tokens=True
        ).strip()
        
        return summary
    
    async def smart_summarize_long_text(
        self, 
        text: str, 
        summary_type: str = "standard"
    ) -> str:
        """
        Умная суммаризация длинного текста с разбивкой на части
        """
        logger.info(f"Запуск умной суммаризации ({summary_type}). Длина: {len(text)}")
        
        # Разбиваем текст на части
        chunks = self._smart_split_text(text, max_chars=get_settings().MAX_CHUNK_SIZE)
        
        if len(chunks) == 1:
            return await self.summarize(chunks[0], summary_type)
        
        # Суммируем каждую часть
        part_summaries = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Суммаризация части {i+1}/{total_chunks}")
            summary = await self.summarize(chunk, summary_type, max_length=1000)
            part_summaries.append(summary)
            await self._light_memory_cleanup()
        
        # Объединяем результаты
        combined_text = "\n\n".join(part_summaries)
        final_summary = await self.summarize(combined_text, summary_type, max_length=6000)
        
        return final_summary
    
    def _smart_split_text(self, text: str, max_chars: int = 14000) -> List[str]:
        """
        Умное разбиение текста на части для обработки
        
        Args:
            text: Исходный текст
            max_chars: Максимальное количество символов в части
            
        Returns:
            Список частей текста
        """
        if len(text) <= max_chars:
            return [text]
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # Если предложение слишком длинное, разбиваем его принудительно
            if sentence_len > max_chars:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Разбиваем длинное предложение
                for i in range(0, sentence_len, max_chars):
                    chunks.append(sentence[i:i + max_chars])
                continue
            
            # Добавляем предложение к текущей части
            if current_length + sentence_len > max_chars:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        # Добавляем последнюю часть
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    async def unload(self) -> None:
        """Выгрузка модели из памяти"""
        if self.model:
            logger.info("Выгрузка Qwen модели из памяти")
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
            
        await self._cleanup_memory()
    
    async def _cleanup_memory(self) -> None:
        """Полная очистка памяти GPU"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"Ошибка очистки памяти: {e}")
    
    async def _light_memory_cleanup(self) -> None:
        """Легкая очистка памяти без выгрузки модели"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Ошибка легкой очистки памяти: {e}")
    
    def is_loaded(self) -> bool:
        """Проверка, загружена ли модель"""
        return self.model is not None
    
    def get_model_info(self) -> Dict:
        """Информация о модели"""
        return {
            "model_path": self.model_path,
            "loaded": self.is_loaded(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "quantization": "4-bit",
        }