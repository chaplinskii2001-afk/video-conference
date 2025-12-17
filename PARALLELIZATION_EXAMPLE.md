# Пример параллелизации Whisper и PyAnnote

## Как это работало раньше (последовательно)

```python
# Временная шкала примерно 60 секунд:
# [Whisper: 0-30s] → [PyAnnote: 30-60s]

audio_path = "conference.wav"

# Шаг 1: Транскрипция (30 секунд)
transcription_segments = await processor.transcribe_audio(audio_path)
# Whisper загружена → обработка → выгрузка

# Шаг 2: Диаризация (30 секунд)  
diarization_segments = await processor.diarize_audio(audio_path)
# PyAnnote загружена → обработка → выгрузка

# Итого: ~60 секунд
```

## Как это работает теперь (параллельно)

```python
# Временная шкала примерно 35 секунд (на 40% быстрее!):
# [Whisper: 0-30s]
# [PyAnnote: 0-35s]  
# └─────────────────┘ (Одновременно!)

audio_path = "conference.wav"

# Параллельная обработка (35 секунд вместо 60)
transcription_segments, diarization_segments = await processor.process_transcription_and_diarization_parallel(audio_path)
# Обе модели загружаются
# ├─ Whisper обрабатывает аудио
# └─ PyAnnote одновременно обрабатывает аудио
# Обе модели выгружаются

# Итого: ~35 секунд (параллельная обработка)
```

## Процесс оркестрации

### Загрузка моделей

```python
# 1. Загружаем Whisper (с параметром skip_unload=True)
success1 = await model_manager.load_whisper(skip_unload=True)

# 2. Загружаем PyAnnote (также с skip_unload=True)
success2 = await model_manager.load_diarization(skip_unload=True)

# Результат: обе модели в памяти GPU одновременно
```

### Параллельное выполнение

```python
# Запускаем обе задачи параллельно через asyncio
transcription_segments, diarization_segments = await asyncio.gather(
    processor._transcribe_audio_parallel(audio_path),
    processor._diarize_audio_parallel(audio_path)
)

# asyncio переключается между задачами:
# - Когда Whisper ждет данных, обрабатывает PyAnnote
# - Когда PyAnnote ждет данных, обрабатывает Whisper
# - На GPU обе модели работают параллельно (если позволяет память)
```

### Выгрузка моделей

```python
# После завершения обе модели выгружаются одновременно
await model_manager.unload_whisper_and_diarization()

# Или просто вызвать:
await model_manager.unload_all_models()  # Выгрузит все модели
```

## Интеграция в основной процесс

### Вызов из process_media()

```python
async def process_media(self, file_path, task_id, media_type="video"):
    # ... подготовка аудио ...
    
    # НОВОЕ: Параллельная обработка вместо последовательной
    transcription_segments, diarization_segments = await self.process_transcription_and_diarization_parallel(audio_path)
    
    # Остальной процесс остается без изменений
    aligned_segments = self.align_transcription_and_diarization(
        transcription_segments,
        diarization_segments
    )
    
    # ... суммаризация, сохранение и т.д. ...
```

## Мониторинг использования GPU

### Пример логов

```
2024-XX-XX Запуск параллельной обработки транскрипции и диаризации
2024-XX-XX Загрузка моделей Whisper и PyAnnote...
2024-XX-XX ✅ Whisper загружен успешно
2024-XX-XX ✅ PyAnnote загружен успешно
2024-XX-XX Обе модели загружены, запуск параллельной обработки...
2024-XX-XX Транскрипция в процессе...
2024-XX-XX Диаризация в процессе...
2024-XX-XX Транскрипция завершена: 1250 сегментов
2024-XX-XX Диаризация завершена: 450 сегментов
2024-XX-XX Параллельная обработка завершена
2024-XX-XX Выгрузка моделей Whisper и PyAnnote
```

### Использование памяти GPU

```
ДО параллелизации:
├─ Whisper: загрузка → использование → выгрузка
├─ PyAnnote: загрузка → использование → выгрузка
└─ Пиковое использование памяти: ~8GB (последовательно)

ПОСЛЕ параллелизации:
├─ Whisper + PyAnnote: загрузка (одновременно) → использование → выгрузка
└─ Пиковое использование памяти: ~10GB (одновременно, примерно +25%)
```

## Обработка ошибок

### Пример 1: Ошибка загрузки Whisper

```python
try:
    await model_manager.load_whisper(skip_unload=True)
    await model_manager.load_diarization(skip_unload=True)
    # ... обработка ...
except Exception as e:
    logger.error(f"Ошибка: {e}")
finally:
    # Гарантированная выгрузка обеих моделей
    await model_manager.unload_whisper_and_diarization()
```

### Пример 2: Ошибка во время параллельной обработки

```python
try:
    result = await asyncio.gather(
        _transcribe_audio_parallel(audio_path),
        _diarize_audio_parallel(audio_path)
    )
except Exception as e:
    logger.error(f"Ошибка параллельной обработки: {e}")
finally:
    # Обе модели все равно выгружаются
    await model_manager.unload_whisper_and_diarization()
```

## Рекомендации

### ✅ Что делать

- ✅ Использовать параллельную обработку для длинных видео (>30 минут)
- ✅ Убедиться, что GPU имеет достаточно памяти (минимум 8GB VRAM)
- ✅ Мониторить использование памяти GPU во время обработки
- ✅ Использовать новый метод `process_transcription_and_diarization_parallel()`

### ❌ Что избежать

- ❌ Не загружайте другие большие модели одновременно с Whisper и PyAnnote
- ❌ Не используйте старые методы `transcribe_audio()` и `diarize_audio()` отдельно в цикле
- ❌ Не пытайтесь вручную управлять выгрузкой моделей (система делает это автоматически)

## Тестирование параллелизма

### Способ 1: Проверка логов

```bash
# Смотрим логи и ищем парных сообщений:
grep "Транскрипция в процессе" app.log
grep "Диаризация в процессе" app.log
# Если они выводятся почти одновременно - параллелизм работает!
```

### Способ 2: Мониторинг GPU

```bash
# В отдельном терминале смотрим использование GPU:
watch -n 1 nvidia-smi

# Во время параллельной обработки должны видеть:
# - Обе модели загружены в память
# - Обе модели используют GPU (не 0%)
# - Использование памяти выше, чем при последовательной обработке
```

### Способ 3: Сравнение времени

```python
import time

# Старый способ (последовательный)
start = time.time()
seg1 = await processor.transcribe_audio(audio_path)
seg2 = await processor.diarize_audio(audio_path)
old_time = time.time() - start

# Новый способ (параллельный)
start = time.time()
seg1, seg2 = await processor.process_transcription_and_diarization_parallel(audio_path)
new_time = time.time() - start

print(f"Старый способ: {old_time:.2f}s")
print(f"Новый способ: {new_time:.2f}s")
print(f"Ускорение: {old_time/new_time:.2f}x раз")
```

## Возможные проблемы и решения

| Проблема | Решение |
|----------|---------|
| `RuntimeError: CUDA out of memory` | Увеличить VRAM GPU или использовать меньшие модели |
| Параллелизм не ускоряет обработку | Возможно, одна модель работает намного быстрее другой |
| Логи показывают последовательное выполнение | Проверить, что используется новый метод `process_transcription_and_diarization_parallel()` |
| Старый код выдает ошибку | Убедиться, что используется параметр `skip_unload=False` (по умолчанию) |

---

Для получения дополнительной информации см.:
- `PARALLELIZATION.md` - Общее описание
- `IMPLEMENTATION_NOTES.md` - Технические деталии
- `CHANGELOG_PARALLELIZATION.md` - Полный список изменений
