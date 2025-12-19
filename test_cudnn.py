#!/usr/bin/env python3
"""
Тест для проверки доступности библиотек cuDNN
"""
import os
import sys
import ctypes
import subprocess

def check_cudnn_libs():
    """Проверка наличия библиотек cuDNN в системе"""
    print("=" * 60)
    print("ПРОВЕРКА БИБЛИОТЕК CUDNN")
    print("=" * 60)
    
    # Проверяем LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    print(f"\n✓ LD_LIBRARY_PATH: {ld_path}")
    
    # Список необходимых библиотек cuDNN 9
    cudnn_libs = [
        'libcudnn.so.9',
        'libcudnn_cnn.so.9',
        'libcudnn_ops.so.9',
        'libcudnn_adv.so.9',
        'libcudnn_graph.so.9',
    ]
    
    print("\n" + "=" * 60)
    print("ПОИСК БИБЛИОТЕК CUDNN:")
    print("=" * 60)
    
    found_libs = []
    missing_libs = []
    
    for lib in cudnn_libs:
        try:
            # Пытаемся загрузить библиотеку
            ctypes.CDLL(lib)
            print(f"✅ {lib}: НАЙДЕНА")
            found_libs.append(lib)
        except OSError as e:
            print(f"❌ {lib}: НЕ НАЙДЕНА")
            missing_libs.append(lib)
            
            # Попробуем найти через ldconfig
            try:
                result = subprocess.run(
                    ['ldconfig', '-p'],
                    capture_output=True,
                    text=True
                )
                if lib.split('.so')[0] in result.stdout:
                    print(f"   (но видна в ldconfig)")
            except Exception:
                pass
    
    # Проверка через find в стандартных путях
    print("\n" + "=" * 60)
    print("ПОИСК В ФАЙЛОВОЙ СИСТЕМЕ:")
    print("=" * 60)
    
    search_paths = [
        '/usr/lib',
        '/usr/local/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/cuda/lib64',
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            try:
                result = subprocess.run(
                    ['find', path, '-name', 'libcudnn*.so*', '-type', 'f'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.stdout.strip():
                    print(f"\n📂 {path}:")
                    for line in result.stdout.strip().split('\n'):
                        print(f"   {line}")
            except Exception as e:
                print(f"⚠️  Ошибка поиска в {path}: {e}")
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ:")
    print("=" * 60)
    print(f"✅ Найдено библиотек: {len(found_libs)}/{len(cudnn_libs)}")
    
    if missing_libs:
        print(f"❌ Отсутствуют библиотеки:")
        for lib in missing_libs:
            print(f"   - {lib}")
        return False
    else:
        print("✅ Все необходимые библиотеки cuDNN найдены!")
        return True

def check_pytorch_cudnn():
    """Проверка поддержки cuDNN в PyTorch"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА PYTORCH + CUDNN:")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch версия: {torch.__version__}")
        print(f"✅ CUDA доступна: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA версия: {torch.version.cuda}")
            print(f"✅ cuDNN версия: {torch.backends.cudnn.version()}")
            print(f"✅ cuDNN включен: {torch.backends.cudnn.enabled}")
            
            # Пробуем создать простой тензор и выполнить операцию
            try:
                x = torch.randn(1, 3, 224, 224).cuda()
                conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
                y = conv(x)
                print(f"✅ Тест Conv2d: УСПЕШНО (output shape: {y.shape})")
                return True
            except Exception as e:
                print(f"❌ Тест Conv2d: ОШИБКА - {e}")
                return False
        else:
            print("⚠️  CUDA недоступна (возможно, запуск вне контейнера)")
            return True
            
    except ImportError as e:
        print(f"❌ PyTorch не установлен: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка проверки PyTorch: {e}")
        return False

if __name__ == "__main__":
    print("\n🔍 ДИАГНОСТИКА CUDNN\n")
    
    libs_ok = check_cudnn_libs()
    torch_ok = check_pytorch_cudnn()
    
    print("\n" + "=" * 60)
    print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
    print("=" * 60)
    
    if libs_ok and torch_ok:
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
        sys.exit(0)
    else:
        print("❌ ОБНАРУЖЕНЫ ПРОБЛЕМЫ!")
        print("\n💡 РЕШЕНИЕ:")
        print("   Установите полный пакет cuDNN:")
        print("   apt-get update && apt-get install -y libcudnn9-cuda-12")
        sys.exit(1)
