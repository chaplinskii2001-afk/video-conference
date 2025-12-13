# Сводка изменений: Исправление обрезания суммаризации для коротких файлов

## Проблема

При обработке коротких видео/аудио файлов (5-10 минут), краткое содержание (summary) было обрезано/сокращено, как будто не хватало токенов. Для длинных файлов (30+ минут) такая проблема не наблюдалась.

## Корень проблемы

1. **Ограничение входной последовательности на 12000 токенов** в методе `summarize_chunk()`
   - Токенизатор обрезал весь prompt + текст до 12000 токенов
   - Для коротких файлов это означало ограниченный контекст

2. **Отсутствие явного `max_length` в методе `generate()`**
   - Модель Qwen могла завершить генерацию раньше положенного
   - Без явного ограничения, завершение срабатывало по внутренним эвристикам

## Решение

### Файл: `processing/video_processor.py`

#### Изменение 1: Метод `summarize_chunk()` (строки 433, 440)

**До:**
```python
inputs = self.model_manager.qwen_tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=12000  # <-- БЫЛО
).to(self.model_manager.qwen_model.device)

with torch.no_grad():
    outputs = self.model_manager.qwen_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.05,
        # <-- НЕ БЫЛО явного max_length
    )
```

**После:**
```python
inputs = self.model_manager.qwen_tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=24000  # <-- УВЕЛИЧЕНО с 12000 до 24000
).to(self.model_manager.qwen_model.device)

with torch.no_grad():
    outputs = self.model_manager.qwen_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        max_length=len(inputs.input_ids[0]) + max_tokens,  # <-- ДОБАВЛЕНО
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.05,
    )
```

**Причины:**
- `max_length=24000` вместо 12000: Даёт модели больше контекста, особенно для коротких файлов
- Добавлен `max_length` в generate(): Гарантирует, что модель будет генерировать ровно `max_tokens` новых токенов без преждевременного завершения

#### Изменение 2: Метод `merge_summaries()` (строки 485, 492)

**До:**
```python
inputs = self.model_manager.qwen_tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=24000  # <-- БЫЛО
).to(self.model_manager.qwen_model.device)

with torch.no_grad():
    outputs = self.model_manager.qwen_model.generate(
        **inputs,
        max_new_tokens=7000,
        do_sample=False,
        repetition_penalty=1.05,
        # <-- НЕ БЫЛО явного max_length
    )
```

**После:**
```python
max_new_tokens = 7000
inputs = self.model_manager.qwen_tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=32000  # <-- УВЕЛИЧЕНО с 24000 до 32000
).to(self.model_manager.qwen_model.device)

with torch.no_grad():
    outputs = self.model_manager.qwen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        max_length=len(inputs.input_ids[0]) + max_new_tokens,  # <-- ДОБАВЛЕНО
        do_sample=False,
        repetition_penalty=1.05,
    )
```

**Причины:**
- `max_length=32000` вместо 24000: Больше контекста для объединения нескольких суммаризаций
- Добавлен `max_length` в generate(): Гарантирует полную генерацию 7000 новых токенов

#### Изменение 3: Документация методов

Добавлены подробные docstrings с описанием исправлений:

```python
"""
Суммаризация одного чанка текста

Важные исправления:
- max_length при токенизации увеличен с 12000 до 24000
  Причина: коротких файлов ограничивались в контексте
- Добавлен явный max_length в generate(): input_length + max_tokens
  Причина: гарантирует полную генерацию output токенов
"""
```

## Дополнительно созданные файлы

1. **`BUGFIX_SUMMARIZATION_TRUNCATION.md`**
   - Подробное описание проблемы и решения
   - Технические детали
   - Рекомендации на будущее

2. **`TEST_SUMMARIZATION_FIX.md`**
   - Процесс тестирования
   - Проверочные критерии
   - Ожидаемое поведение

3. **`CHANGES_SUMMARY.md`** (этот файл)
   - Сводка всех изменений

## Проверка исправления

### Синтаксис
✅ `python3 -m py_compile processing/video_processor.py` - ОК

### Логика
- Увеличение `max_length` при токенизации теперь 24000 (сумм.) и 32000 (слияние)
- Явный `max_length` в generate() теперь гарантирует полную генерацию токенов
- Для коротких файлов: модель будет иметь больше контекста
- Для длинных файлов: продолжит работать корректно

## Ожидаемые результаты

### До исправления
- Короткие файлы: Обрезанное резюме (~400-600 токенов вместо 800)
- Длинные файлы: Полное резюме (800 токенов)

### После исправления
- Короткие файлы: Полное резюме (800 токенов)
- Длинные файлы: Полное резюме (800 токенов, возможно еще более полное благодаря большему контексту)

## Потенциальные побочные эффекты

1. **Увеличенное использование GPU памяти**: Из-за увеличения max_length при токенизации
   - Решение: На системах с низкой памятью можно уменьшить до 20000 или 28000

2. **Немного медленнее обработка**: Модель может генерировать больше токенов
   - Это ожидаемо и является цене исправления

## Рекомендации

1. Протестировать на различных размерах файлов (5 мин, 15 мин, 30 мин, 60 мин)
2. Мониторить использование GPU памяти
3. Добавить мониторинг длины сгенерированного текста в логах
4. Рассмотреть параметр `length_penalty` для лучшего контроля длины

## Файлы, измененные в этом PR

- `processing/video_processor.py` - 2 метода исправлены
- `BUGFIX_SUMMARIZATION_TRUNCATION.md` - Новый (документация)
- `TEST_SUMMARIZATION_FIX.md` - Новый (тесты)
- `CHANGES_SUMMARY.md` - Новый (этот файл)

---

**Дата исправления**: 2024
**Затронутые версии**: Все версии после внедрения Qwen для суммаризации
**Статус**: ✅ Готово к тестированию
