// Упрощенный и читаемый JavaScript для управления UI

let currentTaskId = null;
let progressCheckInterval = null;

// ==================== Переключение табов ====================

function switchTab(tabName) {
    // Скрываем все табы
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Убираем активный класс у кнопок
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Показываем выбранный таб
    if (tabName === 'file') {
        document.getElementById('file-tab').classList.add('active');
        event.target.classList.add('active');
    } else {
        document.getElementById('url-tab').classList.add('active');
        event.target.classList.add('active');
    }
}

// ==================== Обработка файлов ====================

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showFileInfo(file) {
    const fileInfo = document.getElementById('file-info');
    const fileExt = file.name.split('.').pop().toLowerCase();
    const audioFormats = ['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg'];
    const fileType = audioFormats.includes(fileExt) ? 'Аудио' : 'Видео';
    
    fileInfo.innerHTML = `
        <div class="file-info-row">
            <span class="file-info-label">Тип:</span>
            <span class="file-info-value">${fileType}</span>
        </div>
        <div class="file-info-row">
            <span class="file-info-label">Имя файла:</span>
            <span class="file-info-value">${file.name}</span>
        </div>
        <div class="file-info-row">
            <span class="file-info-label">Размер:</span>
            <span class="file-info-value">${formatFileSize(file.size)}</span>
        </div>
    `;
    fileInfo.classList.remove('hidden');
}

// ==================== Начало обработки ====================

async function startProcessing() {
    const fileInput = document.getElementById('file-input');
    const urlInput = document.getElementById('url-input');
    const summaryType = document.getElementById('summary-type').value;
    const processBtn = document.getElementById('process-btn');
    
    // Сброс предыдущих результатов
    hideError();
    hideResults();
    resetProgress();
    
    // Проверка входных данных
    const formData = new FormData();
    let hasInput = false;
    
    if (fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
        hasInput = true;
    } else if (urlInput.value.trim() !== '') {
        formData.append('url', urlInput.value.trim());
        hasInput = true;
    }
    
    if (!hasInput) {
        showError('Пожалуйста, выберите файл или укажите ссылку');
        return;
    }
    
    formData.append('summary_type', summaryType);
    
    // Показываем прогресс и блокируем кнопку
    showProgress();
    processBtn.disabled = true;
    
    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Ошибка сервера');
        }
        
        const data = await response.json();
        currentTaskId = data.task_id;
        
        // Запускаем отслеживание прогресса
        startProgressTracking();
        
    } catch (error) {
        showError(error.message);
        hideProgress();
        processBtn.disabled = false;
    }
}

// ==================== Отслеживание прогресса ====================

function startProgressTracking() {
    if (progressCheckInterval) {
        clearInterval(progressCheckInterval);
    }
    
    // Проверяем прогресс каждые 2 секунды
    progressCheckInterval = setInterval(async () => {
        try {
            const response = await fetch(`/progress/${currentTaskId}`);
            if (!response.ok) return;
            
            const data = await response.json();
            updateProgressDisplay(data);
            
            // Если обработка завершена
            if (data.status === 'completed') {
                clearInterval(progressCheckInterval);
                hideProgress();
                showResults(data);
                document.getElementById('process-btn').disabled = false;
            } else if (data.status === 'error') {
                clearInterval(progressCheckInterval);
                hideProgress();
                showError(data.error || 'Произошла ошибка при обработке');
                document.getElementById('process-btn').disabled = false;
            }
            
        } catch (error) {
            console.error('Ошибка при проверке прогресса:', error);
        }
    }, 2000);
}

// ==================== Обновление UI прогресса ====================

function updateProgressDisplay(data) {
    const percent = data.percent || 0;
    
    // Обновляем прогресс-бар
    document.getElementById('progress-bar-fill').style.width = percent + '%';
    document.getElementById('progress-percentage').textContent = percent + '%';
    
    // Обновляем текущий этап с информацией от сервера
    if (data.current_stage_display) {
        updateCurrentStage(data.current_stage_display);
    } else {
        // Fallback для старых ответов сервера
        updateCurrentStageFallback(data.current_stage);
    }
}

function updateCurrentStage(stageDisplay) {
    // Используем информацию из API сервера
    if (!stageDisplay || typeof stageDisplay !== 'object') {
        return;
    }
    
    const stageName = stageDisplay.display_name || 'Обработка';
    const stageDescription = stageDisplay.description || '';
    const stageIcon = stageDisplay.icon || '⏳';
    
    document.getElementById('current-stage-name').textContent = stageName;
    document.getElementById('stage-description').textContent = stageDescription;
    document.getElementById('stage-icon').textContent = stageIcon;
}

function updateCurrentStageFallback(stage) {
    // Преобразуем stage в понятное название (для совместимости)
    const stageNames = {
        'initialization': 'Начало обработки',
        'download': 'Подготовка файла',
        'preprocessing': 'Подготовка файла',
        'audio_extraction': 'Извлечение аудио',
        'audio_conversion': 'Конвертация аудио',
        'loading_models': 'Загрузка моделей',
        'transcription': 'Транскрипция речи',
        'diarization': 'Определение спикеров',
        'merging': 'Объединение результатов',
        'summarization': 'Создание документа',
        'saving': 'Сохранение результатов',
        'completed': 'Обработка завершена'
    };
    
    const stageIcons = {
        'initialization': '🚀',
        'download': '📥',
        'preprocessing': '⚙️',
        'audio_extraction': '🎵',
        'audio_conversion': '🎵',
        'loading_models': '🤖',
        'transcription': '🎤',
        'diarization': '👥',
        'merging': '🔗',
        'summarization': '📝',
        'saving': '💾',
        'completed': '✅'
    };
    
    const stageName = stageNames[stage] || 'Обработка';
    const stageIcon = stageIcons[stage] || '⏳';
    
    document.getElementById('current-stage-name').textContent = stageName;
    document.getElementById('stage-icon').textContent = stageIcon;
    document.getElementById('stage-description').textContent = '';
}

// ==================== Отображение результатов ====================

function showResults(data) {
    const resultData = data.result || data;
    
    // Обновляем статистику
    document.getElementById('stat-segments').textContent = resultData.segments_count || 0;
    document.getElementById('stat-speakers').textContent = resultData.speakers_count || 0;
    document.getElementById('stat-time').textContent = (resultData.processing_time_minutes || 0) + ' мин';
    
    // Устанавливаем ссылки для скачивания
    const taskId = resultData.task_id || currentTaskId;
    document.getElementById('download-summary').href = `/download/${taskId}/summary`;
    document.getElementById('download-transcription').href = `/download/${taskId}/transcription`;
    
    // Показываем результаты
    document.getElementById('results-section').classList.remove('hidden');
}

function hideResults() {
    document.getElementById('results-section').classList.add('hidden');
}

// ==================== Обработка ошибок ====================

function showError(message) {
    document.getElementById('error-message').textContent = message;
    document.getElementById('error-section').classList.remove('hidden');
    
    // Останавливаем отслеживание прогресса
    if (progressCheckInterval) {
        clearInterval(progressCheckInterval);
        progressCheckInterval = null;
    }
}

function hideError() {
    document.getElementById('error-section').classList.add('hidden');
}

// ==================== Управление прогрессом ====================

function showProgress() {
    document.getElementById('progress-section').classList.remove('hidden');
}

function hideProgress() {
    document.getElementById('progress-section').classList.add('hidden');
}

function resetProgress() {
    document.getElementById('progress-bar-fill').style.width = '0%';
    document.getElementById('progress-percentage').textContent = '0%';
    document.getElementById('current-stage-name').textContent = 'Начало обработки задачи';
    document.getElementById('stage-description').textContent = 'Инициализация системы';
    document.getElementById('stage-icon').textContent = '🚀';
}

// ==================== Инициализация при загрузке ====================

document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file-input');
    const fileUploadZone = document.getElementById('file-upload-zone');
    
    // Обработка выбора файла
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            showFileInfo(this.files[0]);
        }
    });
    
    // Drag and Drop
    fileUploadZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('dragging');
    });
    
    fileUploadZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('dragging');
    });
    
    fileUploadZone.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('dragging');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            showFileInfo(files[0]);
        }
    });
});
