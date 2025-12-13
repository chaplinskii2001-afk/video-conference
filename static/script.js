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
    
    // Обновляем этапы
    updateSteps(data.current_stage);
    
    // Обновляем логи
    updateLogs(data.logs || []);
}

function updateSteps(currentStage) {
    const stageMapping = {
        'download': 'step-download',
        'preprocessing': 'step-download',
        'audio_extraction': 'step-download',
        'transcription': 'step-transcription',
        'loading_models': 'step-transcription',
        'diarization': 'step-diarization',
        'merging': 'step-diarization',
        'summarization': 'step-summarization',
        'saving': 'step-summarization'
    };
    
    const currentStepId = stageMapping[currentStage];
    
    // Сбрасываем все шаги
    document.querySelectorAll('.status-step').forEach(step => {
        step.classList.remove('active', 'completed');
        step.querySelector('.step-icon').textContent = '⏳';
    });
    
    // Отмечаем завершенные и текущий этап
    const steps = ['step-download', 'step-transcription', 'step-diarization', 'step-summarization'];
    let foundCurrent = false;
    
    for (const stepId of steps) {
        const step = document.getElementById(stepId);
        
        if (stepId === currentStepId) {
            step.classList.add('active');
            step.querySelector('.step-icon').textContent = '⚙️';
            foundCurrent = true;
        } else if (!foundCurrent) {
            step.classList.add('completed');
            step.querySelector('.step-icon').textContent = '✅';
        }
    }
}

function updateLogs(logs) {
    const logsContainer = document.getElementById('logs-container');
    logsContainer.innerHTML = '';
    
    // Показываем последние 10 логов
    const recentLogs = logs.slice(-10);
    
    recentLogs.forEach(log => {
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        
        // Парсим лог: [время] сообщение
        const match = log.match(/\[(.*?)\]\s*(.*)/);
        if (match) {
            logEntry.innerHTML = `
                <span class="log-time">[${match[1]}]</span>
                <span class="log-message">${match[2]}</span>
            `;
        } else {
            logEntry.innerHTML = `<span class="log-message">${log}</span>`;
        }
        
        logsContainer.appendChild(logEntry);
    });
    
    // Автоскролл к последнему логу
    logsContainer.scrollTop = logsContainer.scrollHeight;
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
    document.getElementById('logs-container').innerHTML = '';
    
    // Сброс всех этапов
    document.querySelectorAll('.status-step').forEach(step => {
        step.classList.remove('active', 'completed');
        step.querySelector('.step-icon').textContent = '⏳';
    });
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
