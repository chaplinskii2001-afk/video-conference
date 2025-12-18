"""
Справочник этапов обработки файла для отображения пользователю.
Содержит маппинги между техническими названиями и человекопонятными.
"""

# Справочник этапов с полной информацией
PROCESSING_STAGES = {
    # Ожидание
    "queued": {
        "order": 0,
        "display_name": "В очереди",
        "description": "Ожидание начала обработки"
    },

    # Этап 1: Начинаем обрабатывать задачу
    "task_started": {
        "order": 1,
        "display_name": "Начинаем обработку",
        "description": "Инициализация обработки задачи"
    },

    # Этап 2: Загружаем AI модели (Whisper + PyAnnote)
    "loading_ai_models": {
        "order": 2,
        "display_name": "Загружаем AI модели",
        "description": "Подготовка моделей Whisper и PyAnnote"
    },

    # Этап 3: Делаем расшифровку (параллельная обработка)
    "transcribing": {
        "order": 3,
        "display_name": "Делаем расшифровку",
        "description": "Распознавание речи и определение спикеров"
    },

    # Этап 4: Транскрипция завершена
    "transcription_completed": {
        "order": 4,
        "display_name": "Транскрипция завершена",
        "description": "Распознавание речи выполнено"
    },

    # Этап 5: Диаризация завершена
    "diarization_completed": {
        "order": 5,
        "display_name": "Диаризация завершена",
        "description": "Определение спикеров выполнено"
    },

    # Этап 6: Загружаем Qwen
    "loading_qwen": {
        "order": 6,
        "display_name": "Загружаем модель суммаризации",
        "description": "Подготовка модели Qwen"
    },

    # Этап 7: Делаем краткое содержание
    "summarizing": {
        "order": 7,
        "display_name": "Делаем краткое содержание",
        "description": "Создание документа с итогами"
    },

    # Этап 8: Все готово
    "task_completed": {
        "order": 8,
        "display_name": "Все готово",
        "description": "Обработка успешно завершена"
    },

    # Устаревшие этапы (для обратной совместимости)
    "initialization": {
        "order": 1,
        "display_name": "Начало обработки задачи",
        "description": "Инициализация системы"
    },
    "download": {
        "order": 1,
        "display_name": "Подготовка файла",
        "description": "Скачивание или сохранение файла"
    },
    "preprocessing": {
        "order": 1,
        "display_name": "Подготовка файла",
        "description": "Подготовка к обработке"
    },
    "audio_extraction": {
        "order": 1,
        "display_name": "Извлечение аудиодорожки",
        "description": "Извлечение звука из видео"
    },
    "audio_conversion": {
        "order": 1,
        "display_name": "Конвертация аудио",
        "description": "Подготовка аудиофайла"
    },
    "loading_models": {
        "order": 2,
        "display_name": "Загрузка моделей ИИ",
        "description": "Подготовка нейросетей"
    },
    "transcription_and_diarization": {
        "order": 3,
        "display_name": "Параллельная обработка",
        "description": "Распознавание речи (Whisper) и определение спикеров (PyAnnote) одновременно"
    },
    "transcription": {
        "order": 3,
        "display_name": "Транскрипция речи",
        "description": "Распознавание речи (Whisper)"
    },
    "diarization": {
        "order": 3,
        "display_name": "Определение спикеров",
        "description": "Разделение по говорящим людям (PyAnnote)"
    },
    "merging": {
        "order": 5,
        "display_name": "Объединение результатов",
        "description": "Согласование транскрипции и спикеров"
    },
    "summarization": {
        "order": 7,
        "display_name": "Создание документа",
        "description": "Написание краткого содержания (Qwen)"
    },
    "saving": {
        "order": 7,
        "display_name": "Сохранение результатов",
        "description": "Финализация и сохранение"
    },
    "completed": {
        "order": 8,
        "display_name": "Обработка завершена",
        "description": "Успешно завершено"
    }
}


def get_display_stage(technical_stage: str) -> dict:
    """
    Получить информацию о этапе для отображения пользователю
    
    Args:
        technical_stage: техническое имя этапа
        
    Returns:
        dict с display_name, description и order
    """
    if technical_stage in PROCESSING_STAGES:
        return PROCESSING_STAGES[technical_stage]
    
    # Fallback для неизвестных этапов
    return {
        "order": 0,
        "display_name": "Обработка",
        "description": technical_stage
    }


def get_stage_display_name(technical_stage: str) -> str:
    """Получить отображаемое имя этапа"""
    stage_info = get_display_stage(technical_stage)
    return stage_info["display_name"]


def get_stage_order(technical_stage: str) -> int:
    """Получить порядковый номер этапа для прогресса"""
    stage_info = get_display_stage(technical_stage)
    return stage_info["order"]
