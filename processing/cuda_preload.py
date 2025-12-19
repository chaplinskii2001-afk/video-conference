import ctypes
import glob
import importlib.util
import logging
import os
from typing import Optional, Sequence, Tuple, List

_PRELOADED = False
_PRELOAD_OK = False
_PRELOADED_VERSION: Optional[int] = None


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    return logger or logging.getLogger(__name__)


def _find_package_lib_dir(module_name: str, *, subdir: str = "lib") -> Optional[str]:
    spec = importlib.util.find_spec(module_name)
    if not spec or not spec.origin:
        return None

    pkg_dir = os.path.dirname(spec.origin)
    lib_dir = os.path.join(pkg_dir, subdir)
    if os.path.isdir(lib_dir):
        return lib_dir

    return None


def _candidate_lib_dirs() -> List[str]:
    # В новых torch wheels cuDNN часто поставляется как отдельный пакет nvidia-cudnn-cu12
    # и лежит в site-packages/nvidia/cudnn/lib.
    dirs: List[str] = []

    cudnn_dir = _find_package_lib_dir("nvidia.cudnn")
    if cudnn_dir:
        dirs.append(cudnn_dir)

    torch_dir = _find_package_lib_dir("torch")
    if torch_dir:
        dirs.append(torch_dir)

    # Убираем дубликаты, сохраняя порядок
    seen = set()
    unique: List[str] = []
    for d in dirs:
        if d not in seen:
            unique.append(d)
            seen.add(d)

    return unique


def _load_library(path: str) -> None:
    # RTLD_GLOBAL нужен, чтобы другие модули (ctranslate2/torch) могли переиспользовать уже загруженную cuDNN
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def _try_get_cudnn_version_from_loaded() -> Optional[int]:
    try:
        libcudnn = ctypes.CDLL("libcudnn.so.9")
        libcudnn.cudnnGetVersion.restype = ctypes.c_size_t
        return int(libcudnn.cudnnGetVersion())
    except Exception:
        return None


def _format_cudnn_version(version: int) -> str:
    # cudnnGetVersion возвращает число вида 91002 (9.10.2)
    major = version // 10000
    minor = (version % 10000) // 100
    patch = version % 100
    return f"{major}.{minor}.{patch}"


def preload_cudnn(*, logger: Optional[logging.Logger] = None) -> Tuple[bool, Optional[int]]:
    """Пытается гарантировать, что в процессе загружена cuDNN из Python wheels (а не системная).

    Проблема из логов: PyTorch ожидает cuDNN 9.10.x, но в runtime окружении может
    присутствовать системная cuDNN 9.5.x (например, из CUDA *-cudnn-runtime образа).
    Если первой загрузится системная cuDNN (часто через ctranslate2/faster-whisper),
    PyTorch падает с `cuDNN version incompatibility`.

    Решение: заранее pre-load cuDNN из site-packages (nvidia-cudnn / torch/lib) с RTLD_GLOBAL.

    Возвращает: (успех, версия cudnnGetVersion если удалось определить)
    """

    global _PRELOADED, _PRELOAD_OK, _PRELOADED_VERSION

    log = _get_logger(logger)

    if _PRELOADED:
        return _PRELOAD_OK, _PRELOADED_VERSION

    _PRELOADED = True

    lib_dirs = _candidate_lib_dirs()
    if not lib_dirs:
        log.info("ℹ️ cuDNN preload: не нашли директории с cuDNN в site-packages (nvidia.cudnn/lib или torch/lib)")
        _PRELOAD_OK = False
        _PRELOADED_VERSION = None
        return _PRELOAD_OK, _PRELOADED_VERSION

    # Загружаем наиболее важные библиотеки. На практике достаточно libcudnn.so.9 + (ops/cnn),
    # чтобы и ctranslate2, и torch использовали одинаковую версию.
    lib_name_patterns: Sequence[str] = (
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_graph.so.9",
    )

    loaded_any = False
    for lib_dir in lib_dirs:
        for name in lib_name_patterns:
            path = os.path.join(lib_dir, name)
            if not os.path.exists(path):
                continue

            try:
                _load_library(path)
                loaded_any = True
            except OSError as e:
                # Не падаем — часть библиотек может отсутствовать в конкретной сборке.
                log.debug(f"cuDNN preload: не удалось загрузить {path}: {e}")

        # Дополнительно пытаемся загрузить любые libcudnn*.so.9* (на случай нестандартных имен)
        for path in sorted(glob.glob(os.path.join(lib_dir, "libcudnn*.so.9*"))):
            if not os.path.isfile(path):
                continue
            try:
                _load_library(path)
                loaded_any = True
            except OSError:
                continue

    version = _try_get_cudnn_version_from_loaded()

    if loaded_any:
        _PRELOAD_OK = True
        _PRELOADED_VERSION = version
        if version is not None:
            log.info(f"✅ cuDNN preload: загружена cuDNN {_format_cudnn_version(version)} (RTLD_GLOBAL)")
        else:
            log.info("✅ cuDNN preload: cuDNN загружена (версию определить не удалось)")
    else:
        _PRELOAD_OK = False
        _PRELOADED_VERSION = version
        log.info("ℹ️ cuDNN preload: подходящие cuDNN библиотеки не найдены/не загружены")

    return _PRELOAD_OK, _PRELOADED_VERSION
