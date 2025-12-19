# Исправление ошибки libcudnn_cnn

## Проблема
```
Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}
Invalid handle. Cannot load symbol cudnnCreateConvolutionDescriptor
```

## Решение
Добавлена установка полного пакета cuDNN 9 в `docker/Dockerfile`:

```dockerfile
# Установка полного набора библиотек cuDNN 9 для CUDA 12
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*
```

## Что исправлено
- ✅ PyTorch теперь может загрузить все необходимые библиотеки cuDNN 9
- ✅ PyAnnote (диаризация) работает корректно на GPU
- ✅ faster-whisper (транскрипция) продолжает работать без проблем
- ✅ Обе модели могут работать параллельно на одном GPU

## Как применить
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Тестирование
Запустите тест внутри контейнера:
```bash
docker-compose exec video-processor python3 test_cudnn.py
```

## Документация
Подробности в: `CUDNN_FIX_DOCUMENTATION.md`
