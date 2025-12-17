# Реализация параллелизации Whisper и PyAnnote

## Обзор

Это документ описывает техническую реализацию параллелизации процессов Whisper (транскрипция) и PyAnnote (диаризация) в системе обработки видеоконференций.

## Проблема и решение

### Проблема (до параллелизации)
```
Временная шкала (последовательная обработка):
|----Whisper----|----PyAnnote----|
←───────────────────────────────→ Общее время
```

### Решение (параллельная обработка)
```
Временная шкала (параллельная обработка):
|----Whisper----|
|----PyAnnote----|
←─────────────→ Общее время (сокращено)
```

## Архитектурные изменения

### 1. ModelManager - Управление загрузкой моделей

**Перед**: При загрузке новой модели выгружалась текущая модель
```python
async def load_whisper(self):
    await self.unload_current_model()  # Выгрузка предыдущей модели
    # ... загрузка Whisper
    self.current_loaded_model = "whisper"
```

**После**: Параметр `skip_unload` позволяет удерживать модели в памяти
```python
async def load_whisper(self, skip_unload: bool = False):
    if not skip_unload:
        await self.unload_current_model()
    # ... загрузка Whisper
    if not skip_unload:
        self.current_loaded_model = "whisper"
```

### 2. VideoProcessor - Параллельное выполнение

**Новый метод `process_transcription_and_diarization_parallel()`**:
```python
async def process_transcription_and_diarization_parallel(self, audio_path):
    # 1. Загружаем обе модели
    await self.model_manager.load_whisper(skip_unload=True)
    await self.model_manager.load_diarization(skip_unload=True)
    
    # 2. Запускаем обработку ПАРАЛЛЕЛЬНО
    transcription, diarization = await asyncio.gather(
        self._transcribe_audio_parallel(audio_path),
        self._diarize_audio_parallel(audio_path)
    )
    
    # 3. Выгружаем обе модели
    await self.model_manager.unload_whisper_and_diarization()
    
    return transcription, diarization
```

## Управление памятью

### Сценарий загрузки памяти

1. **Загрузка Whisper** (skip_unload=True)
   - Модель загружается на GPU
   - `current_loaded_model` остается None
   - Запоминаем: Whisper в памяти

2. **Загрузка PyAnnote** (skip_unload=True)
   - Модель загружается на GPU
   - `current_loaded_model` остается None
   - Запоминаем: Whisper и PyAnnote в памяти

3. **Параллельная обработка**
   - Оба процесса работают одновременно
   - asyncio переключается между ними при I/O операциях
   - GPU обрабатывает обе модели (если хватает памяти)

4. **Выгрузка обеих моделей**
   - Метод `unload_whisper_and_diarization()` удаляет обе модели

## Обработка ошибок

Система гарантирует выгрузку моделей даже при ошибках:

```python
try:
    # Загрузка
    await self.model_manager.load_whisper(skip_unload=True)
    await self.model_manager.load_diarization(skip_unload=True)
    
    # Параллельная обработка
    result = await asyncio.gather(transcribe(), diarize())
    
except Exception as e:
    logger.error(f"Ошибка: {e}")
    raise
finally:
    # Гарантированная выгрузка
    await self.model_manager.unload_whisper_and_diarization()
```

## Обратная совместимость

Методы `transcribe_audio()` и `diarize_audio()` остаются функциональны и могут использоваться независимо:

```python
# Старый способ (последовательный)
segments1 = await processor.transcribe_audio(audio_path)
segments2 = await processor.diarize_audio(audio_path)

# Новый способ (параллельный)
segments1, segments2 = await processor.process_transcription_and_diarization_parallel(audio_path)
```

## Метрики производительности

### Ожидаемые улучшения

На современных GPU (NVIDIA RTX 3090 и выше):
- Время на этапе "Транскрипция + Диаризация" может сократиться на **40-60%**
- Увеличение потребления памяти на **15-30%** (обе модели в памяти одновременно)
- Общее время обработки сокращается на **20-30%**

### Зависимости

- Доступная GPU память должна вмещать обе модели одновременно
- Если памяти недостаточно, система выведет ошибку при загрузке

## Логирование и мониторинг

Система логирует все этапы:

```
Запуск параллельной обработки транскрипции и диаризации
Загрузка моделей Whisper и PyAnnote...
Whisper загружен успешно
PyAnnote загружен успешно
Обе модели загружены, запуск параллельной обработки...
Транскрипция завершена: N сегментов
Диаризация завершена: M сегментов
Параллельная обработка завершена
Выгрузка моделей Whisper и PyAnnote
```

## Тестирование

Параллелизм можно тестировать путем:

1. Загрузки большого видеофайла (>1 часа)
2. Отслеживания использования GPU через `nvidia-smi`
3. Проверки того, что обе модели работают одновременно
4. Сравнения времени выполнения до и после параллелизации

## Возможные улучшения в будущем

1. **Асинхронная загрузка моделей**: Можно загружать модели в параллель через asyncio
2. **Адаптивное управление памятью**: Автоматически определять, можно ли одновременно загрузить обе модели
3. **Кэширование моделей**: Сохранять модели в памяти между запросами для еще большего ускорения
4. **Распределенная обработка**: Использовать несколько GPU для разных моделей
