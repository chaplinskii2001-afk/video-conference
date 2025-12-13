"""
Справочник этапов обработки файла для отображения пользователю.
Содержит маппинги между техническими названиями и человекопонятными.
"""

# Справочник этапов с полной информацией
PROCESSING_STAGES = {
    # Инициализация и подготовка
    "initialization": {
        "order": 1,
        "display_name": "Начало обработки задачи",
        "description": "Инициализация системы",
        "icon": "🚀"
    },
    "download": {
        "order": 2,
        "display_name": "Подготовка файла",
        "description": "Скачивание или сохранение файла",
        "icon": "📥"
    },
    "preprocessing": {
        "order": 3,
        "display_name": "Подготовка файла",
        "description": "Подготовка к обработке",
        "icon": "⚙️"
    },
    
    # Работа с аудио
    "audio_extraction": {
        "order": 4,
        "display_name": "Извлечение аудиодорожки",
        "description": "Извлечение звука из видео",
        "icon": "🎵"
    },
    "audio_conversion": {
        "order": 4,
        "display_name": "Конвертация аудио",
        "description": "Подготовка аудиофайла",
        "icon": "🎵"
    },
    
    # Загрузка моделей
    "loading_models": {
        "order": 5,
        "display_name": "Загрузка моделей ИИ",
        "description": "Подготовка нейросетей",
        "icon": "🤖"
    },
    
    # Транскрипция
    "transcription": {
        "order": 6,
        "display_name": "Транскрипция речи",
        "description": "Распознавание речи (Whisper)",
        "icon": "🎤"
    },
    
    # Диаризация
    "diarization": {
        "order": 7,
        "display_name": "Определение спикеров",
        "description": "Разделение по говорящим людям (PyAnnote)",
        "icon": "👥"
    },
    
    # Объединение результатов
    "merging": {
        "order": 8,
        "display_name": "Объединение результатов",
        "description": "Согласование транскрипции и спикеров",
        "icon": "🔗"
    },
    
    # Суммаризация
    "summarization": {
        "order": 9,
        "display_name": "Создание документа",
        "description": "Написание краткого содержания (Qwen)",
        "icon": "📝"
    },
    
    # Сохранение
    "saving": {
        "order": 10,
        "display_name": "Сохранение результатов",
        "description": "Финализация и сохранение",
        "icon": "💾"
    },
    
    # Завершение
    "completed": {
        "order": 11,
        "display_name": "Обработка завершена",
        "description": "Успешно завершено",
        "icon": "✅"
    }
}


def get_display_stage(technical_stage: str) -> dict:
    """
    Получить информацию о этапе для отображения пользователю
    
    Args:
        technical_stage: техническое имя этапа
        
    Returns:
        dict с display_name, description, icon и order
    """
    if technical_stage in PROCESSING_STAGES:
        return PROCESSING_STAGES[technical_stage]
    
    # Fallback для неизвестных этапов
    return {
        "order": 0,
        "display_name": "Обработка",
        "description": technical_stage,
        "icon": "⏳"
    }


def get_stage_display_name(technical_stage: str) -> str:
    """Получить отображаемое имя этапа"""
    stage_info = get_display_stage(technical_stage)
    return stage_info["display_name"]


def get_stage_order(technical_stage: str) -> int:
    """Получить порядковый номер этапа для прогресса"""
    stage_info = get_display_stage(technical_stage)
    return stage_info["order"]
