#!/usr/bin/env python3
"""
Проверка исправления race condition в параллелизации
Проверяет, что whisper_transcribe() теперь проверяет только наличие pipeline
"""

import ast
import sys

def check_whisper_transcribe_logic():
    """Проверяет логику проверки в методе whisper_transcribe"""
    with open("processing/model_manager.py", 'r') as f:
        content = f.read()
        tree = ast.parse(content)
    
    # Находим метод whisper_transcribe
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "whisper_transcribe":
                    # Проверяем первую строку тела функции (проверка условия)
                    if_statement = item.body[0]
                    if isinstance(if_statement, ast.If):
                        # Получаем условие if
                        condition = if_statement.test
                        
                        # Проверяем, что это только проверка whisper_pipeline is None
                        # (не должно быть BoolOp с 'or' и проверкой current_loaded_model)
                        if isinstance(condition, ast.BoolOp):
                            print("❌ ОШИБКА: whisper_transcribe() содержит составное условие (BoolOp)")
                            print("   Это может быть проверка current_loaded_model, которая вызывает race condition")
                            return False
                        
                        # Должна быть простая проверка Compare: self.whisper_pipeline is None
                        if isinstance(condition, ast.Compare):
                            # Проверяем, что это сравнение с is/is not None
                            if any(isinstance(op, (ast.Is, ast.IsNot)) for op in condition.ops):
                                print("✅ whisper_transcribe() использует простую проверку pipeline")
                                print("   Проверяется только наличие self.whisper_pipeline")
                                return True
                        
                        print("❌ ОШИБКА: Неожиданная структура условия в whisper_transcribe()")
                        return False
    
    print("❌ ОШИБКА: Метод whisper_transcribe() не найден")
    return False

def check_parallel_loading_comments():
    """Проверяет комментарии о параллельной загрузке"""
    with open("processing/model_manager.py", 'r') as f:
        content = f.read()
    
    # Проверяем, что есть упоминание skip_unload в load_whisper
    if "skip_unload" in content:
        print("✅ Параметр skip_unload присутствует в коде")
        return True
    else:
        print("⚠️ Параметр skip_unload не найден")
        return False

def main():
    print("=" * 70)
    print("Проверка исправления race condition в параллельной обработке")
    print("=" * 70)
    
    print("\n[1/2] Проверка логики whisper_transcribe()...")
    result1 = check_whisper_transcribe_logic()
    
    print("\n[2/2] Проверка наличия параметра skip_unload...")
    result2 = check_parallel_loading_comments()
    
    print("\n" + "=" * 70)
    if result1 and result2:
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
        print("\nИсправление:")
        print("  • Удалена проверка current_loaded_model из whisper_transcribe()")
        print("  • Теперь проверяется только наличие self.whisper_pipeline")
        print("  • Это позволяет методу работать как с последовательной, так и с")
        print("    параллельной загрузкой моделей")
        return 0
    else:
        print("❌ ПРОВЕРКИ НЕ ПРОЙДЕНЫ")
        return 1

if __name__ == "__main__":
    sys.exit(main())
