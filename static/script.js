let currentTaskId = null;
let progressInterval = null;

function openTab(tabName) {
    // Скрыть все табы
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    // Убрать активный класс у всех кнопок
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    // Показать выбранный таб
    document.getElementById(tabName).classList.add('active');
    event.currentTarget.classList.add('active');
}

async function processVideo() {
    const fileInput = document.getElementById('file-input');
    const urlInput = document.getElementById('url-input');
    const summaryTypeSelect = document.getElementById('summary-type');
    const processBtn = document.getElementById('process-btn');
    const progress = document.getElementById('progress');
    const results = document.getElementById('results');
    const error = document.getElementById('error');

    // Сброс состояний
    error.classList.add('hidden');
    results.classList.add('hidden');
    resetProgress();

    const formData = new FormData();
    let hasInput = false;

    if (fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
        hasInput = true;
    } else if (urlInput.value.trim() !== '') {
        formData.append('url', urlInput.value.trim());
        hasInput = true;
    }
    formData.append('summary_type', summaryTypeSelect.value);

    if (!hasInput) {
        showError('Пожалуйста, выберите файл или укажите ссылку');
        return;
    }

    // Показать прогресс
    progress.classList.remove('hidden');
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

        // Начать отслеживание прогресса
        startProgressTracking(currentTaskId);
    } catch (err) {
        showError(err.message);
        progress.classList.add('hidden');
        processBtn.disabled = false;
    }
}

function startProgressTracking(taskId) {
    if (progressInterval) {
        clearInterval(progressInterval);
    }

    progressInterval = setInterval(async () => {
        try {
            const response = await fetch(`/progress/${taskId}`);
            if (!response.ok) return;

            const progressData = await response.json();
            updateProgress(progressData);

            // Если задача завершена
            if (progressData.status === 'completed' || progressData.status === 'error') {
                clearInterval(progressInterval);

                if (progressData.status === 'completed') {
                    showResults(progressData);
                } else {
                    showError(progressData.error || 'Произошла ошибка при обработке');
                }

                document.getElementById('process-btn').disabled = false;
            }
        } catch (error) {
            console.error('Ошибка при запросе прогресса:', error);
        }
    }, 2000); // Опрашиваем каждые 2 секунды
}

function updateProgress(progressData) {
    // Обновляем прогресс-бар
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const percent = progressData.percent || 0;
    progressFill.style.width = `${percent}%`;
    progressText.textContent = `${percent}%`;

    // Обновляем стадии
    updateStages(progressData.current_stage);

    // Обновляем логи
    updateLogs(progressData.logs || []);
}

function updateStages(currentStage) {
    const stages = {
        'download': 'stage-download',
        'preprocessing': 'stage-download',
        'audio_extraction': 'stage-download',
        'transcription': 'stage-transcription',
        'diarization': 'stage-diarization',
        'merging': 'stage-diarization',
        'summarization': 'stage-summarization',
        'formatting': 'stage-summarization',
        'saving': 'stage-summarization'
    };

    // Сброс всех стадий
    Object.values(stages).forEach(stageId => {
        const stage = document.getElementById(stageId);
        stage.classList.remove('active', 'completed');
        stage.querySelector('.stage-icon').textContent = '⏳';
    });

    // Активация текущих и завершенных стадий
    let foundCurrent = false;
    for (const [stageName, stageId] of Object.entries(stages)) {
        const stage = document.getElementById(stageId);
        if (!foundCurrent) {
            stage.classList.add('completed');
            stage.querySelector('.stage-icon').textContent = '✅';
        }
        if (stageName === currentStage) {
            stage.classList.add('active');
            stage.querySelector('.stage-icon').textContent = '🔄';
            foundCurrent = true;
        }
    }
}

function updateLogs(logs) {
    const logsContainer = document.getElementById('logs');
    logsContainer.innerHTML = '';

    // Показываем только последние 10 логов
    const recentLogs = logs.slice(-10);
    recentLogs.forEach(log => {
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';

        // Извлекаем время и сообщение из лога
        const logMatch = log.match(/\[(.*?)\]\s*(.*)/);
        if (logMatch) {
            const timestamp = logMatch[1];
            const message = logMatch[2];
            logEntry.innerHTML = `
                <span class="log-time">[${timestamp}]</span>
                <span class="log-message">${message}</span>
            `;
        } else {
            logEntry.innerHTML = `<span class="log-message">${log}</span>`;
        }
        logsContainer.appendChild(logEntry);
    });

    // Автоскролл к последнему логу
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

function resetProgress() {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const logsContainer = document.getElementById('logs');

    progressFill.style.width = '0%';
    progressText.textContent = '0%';
    logsContainer.innerHTML = '';

    // Сброс стадий
    document.querySelectorAll('.stage').forEach(stage => {
        stage.classList.remove('active', 'completed');
        stage.querySelector('.stage-icon').textContent = '⏳';
    });
}

function showResults(data) {
    const results = document.getElementById('results');
    const resultStats = document.getElementById('result-stats');
    const summaryLink = document.getElementById('summary-link');
    const transcriptionLink = document.getElementById('transcription-link');
    const progress = document.getElementById('progress');

    // Обновляем статистику
    const resultData = data.result || data;
    resultStats.innerHTML = `
        <div class="stat-item">
            <span class="stat-label">Обработано сегментов:</span>
            <span class="stat-value">${resultData.segments_count || 0}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Обнаружено спикеров:</span>
            <span class="stat-value">${resultData.speakers_count || 0}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Время обработки:</span>
            <span class="stat-value">${resultData.processing_time_minutes || 0} мин.</span>
        </div>
    `;

    // Обновляем ссылки для скачивания
    summaryLink.href = `/download/${resultData.task_id}/summary`;
    transcriptionLink.href = `/download/${resultData.task_id}/transcription`;

    // Показываем кнопки скачивания
    summaryLink.classList.remove('hidden');
    transcriptionLink.classList.remove('hidden');

    // Показываем результаты
    progress.classList.add('hidden');
    results.classList.remove('hidden');
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');

    // Останавливаем отслеживание прогресса
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
}

// Обработчик drag and drop для файлов
document.addEventListener('DOMContentLoaded', function () {
    const fileLabel = document.querySelector('.file-label');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');

    fileLabel.addEventListener('dragover', function (e) {
        e.preventDefault();
        this.style.borderColor = '#007bff';
        this.style.background = '#f8f9ff';
    });

    fileLabel.addEventListener('dragleave', function (e) {
        e.preventDefault();
        this.style.borderColor = '#dee2e6';
        this.style.background = '';
    });

    fileLabel.addEventListener('drop', function (e) {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            updateFileInfo(files[0]);
        }
    });

    fileInput.addEventListener('change', function () {
        if (this.files.length > 0) {
            updateFileInfo(this.files[0]);
        }
    });

    function updateFileInfo(file) {
        const fileExt = file.name.split('.').pop().toLowerCase();
        const isAudio = ['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg'].includes(fileExt);
        const fileType = isAudio ? 'Аудио' : 'Видео';
        fileInfo.innerHTML = `
            <div class="file-type">Тип: ${fileType}</div>
            <div class="file-name">Файл: ${file.name}</div>
            <div class="file-size">Размер: ${formatFileSize(file.size)}</div>
        `;
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
});
