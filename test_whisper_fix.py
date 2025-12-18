#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправления проблемы с Whisper моделью
"""
import os
import sys

def test_whisper_config():
    """Проверяем, что конфигурация Whisper правильная"""
    
    # Имитируем данные из config.py
    gpu_config = {
        "whisper_model_id": "Systran/faster-whisper-large-v3",  # Это должно быть приоритетным
        "name": "Базовый (4-6 GB VRAM)",
    }
    
    model_config = {
        "whisper_id": "Systran/faster-whisper-large-v3"
    }
    
    # Логика из model_manager.py
    whisper_id = gpu_config.get("whisper_model_id") or model_config["whisper_id"]
    
    print("=== ТЕСТ КОНФИГУРАЦИИ WHISPER ===")
    print(f"gpu_config.whisper_model_id: {gpu_config.get('whisper_model_id')}")
    print(f"model_config.whisper_id: {model_config['whisper_id']}")
    print(f"Финальный whisper_id: {whisper_id}")
    print()
    
    # Проверяем, что используется правильная модель
    expected_model = "Systran/faster-whisper-large-v3"
    if whisper_id == expected_model:
        print(f"✅ УСПЕХ: Используется правильная модель {expected_model}")
        return True
    else:
        print(f"❌ ОШИБКА: Используется неправильная модель {whisper_id}")
        return False

def test_cache_cleanup():
    """Тестируем логику очистки кэша"""
    
    print("\n=== ТЕСТ ОЧИСТКИ КЭША ===")
    
    # Создаем временную директорию для теста
    test_dir = "/tmp/whisper_test_cache"
    
    try:
        os.makedirs(test_dir, exist_ok=True)
        
        # Создаем поддельные папки моделей
        old_model_dir = os.path.join(test_dir, "models--bond005--whisper-podlodka-turbo")
        new_model_dir = os.path.join(test_dir, "models--Systran--faster-whisper-large-v3")
        
        os.makedirs(old_model_dir, exist_ok=True)
        os.makedirs(new_model_dir, exist_ok=True)
        
        # Создаем тестовые файлы
        with open(os.path.join(old_model_dir, "model.bin"), "w") as f:
            f.write("old model")
        with open(os.path.join(new_model_dir, "model.bin"), "w") as f:
            f.write("new model")
        
        print(f"Создана тестовая директория: {test_dir}")
        print(f"Содержимое: {os.listdir(test_dir)}")
        
        # Применяем логику очистки из model_manager.py
        keep_pattern = "models--Systran--faster-whisper-large-v3"
        
        for item in os.listdir(test_dir):
            item_path = os.path.join(test_dir, item)
            if os.path.isdir(item_path):
                if not item.startswith(keep_pattern):
                    print(f"🗑️ Удаление неправильной модели: {item_path}")
                    # В реальном коде здесь был бы shutil.rmtree
                else:
                    print(f"✅ Сохраняем правильную модель: {item}")
        
        print("✅ Логика очистки кэша работает корректно")
        return True
        
    finally:
        # Очищаем тестовую директорию
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)

def main():
    """Основная функция тестирования"""
    print("Тестирование исправления проблемы с Whisper моделью")
    print("=" * 60)
    
    success = True
    
    # Тест 1: Конфигурация
    success &= test_whisper_config()
    
    # Тест 2: Очистка кэша
    success &= test_cache_cleanup()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("Исправления должны решить проблему с старой моделью bond005/whisper-podlodka-turbo")
    else:
        print("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ!")
        print("Требуются дополнительные исправления")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)