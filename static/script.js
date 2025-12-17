let clientId = null;
let currentBatchId = null;
let progressCheckInterval = null;
let selectedFiles = [];

const STORAGE_KEYS = {
    clientId: 'vcp_client_id',
    activeBatchId: 'vcp_active_batch_id',
    completedTasks: 'vcp_completed_tasks'
};

function generateUUIDv4() {
    // Fallback if crypto.randomUUID is not available
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function ensureClientId() {
    const existing = localStorage.getItem(STORAGE_KEYS.clientId);
    if (existing) {
        clientId = existing;
        return;
    }

    clientId = (window.crypto && window.crypto.randomUUID) ? window.crypto.randomUUID() : generateUUIDv4();
    localStorage.setItem(STORAGE_KEYS.clientId, clientId);
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function isAudioFile(fileName) {
    const ext = (fileName.split('.').pop() || '').toLowerCase();
    return ['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg'].includes(ext);
}

function getCompletedTasks() {
    try {
        const raw = localStorage.getItem(STORAGE_KEYS.completedTasks);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
}

function setCompletedTasks(tasks) {
    localStorage.setItem(STORAGE_KEYS.completedTasks, JSON.stringify(tasks));
}

function addCompletedTask(task) {
    const tasks = getCompletedTasks();

    if (tasks.some(t => t.task_id === task.task_id)) {
        return;
    }

    tasks.unshift(task);
    setCompletedTasks(tasks.slice(0, 100));
}

function renderCompletedResults() {
    const container = document.getElementById('completed-results');
    const tasks = getCompletedTasks();

    if (!tasks.length) {
        container.innerHTML = '<div class="completed-empty">Пока нет готовых файлов</div>';
        return;
    }

    container.innerHTML = tasks.map(t => {
        const safeName = escapeHtml(t.file_name || t.task_id);
        const summaryUrl = `/download/${encodeURIComponent(t.task_id)}/summary?client_id=${encodeURIComponent(clientId)}`;
        const transcriptionUrl = `/download/${encodeURIComponent(t.task_id)}/transcription?client_id=${encodeURIComponent(clientId)}`;

        return `
            <div class="completed-item">
                <div class="completed-name">${safeName}</div>
                <div class="completed-links">
                    <a class="download-link" href="${summaryUrl}" target="_blank" rel="noopener">Краткое содержание</a>
                    <a class="download-link secondary" href="${transcriptionUrl}" target="_blank" rel="noopener">Расшифровка</a>
                </div>
            </div>
        `;
    }).join('');
}

function clearHistory() {
    localStorage.removeItem(STORAGE_KEYS.completedTasks);
    renderCompletedResults();
}

function switchTab(tabName, event) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-button').forEach(button => button.classList.remove('active'));

    if (tabName === 'file') {
        document.getElementById('file-tab').classList.add('active');
    } else {
        document.getElementById('url-tab').classList.add('active');
    }

    if (event && event.target) {
        event.target.classList.add('active');
    }
}

function setFileInputFiles(files) {
    const fileInput = document.getElementById('file-input');
    const dataTransfer = new DataTransfer();
    files.forEach(file => dataTransfer.items.add(file));
    fileInput.files = dataTransfer.files;
}

function onFilesSelected(files) {
    const newFiles = Array.from(files);
    // Filter duplicates by name if needed, or just append
    // Here we just append all selected files to the existing list
    selectedFiles = selectedFiles.concat(newFiles);
    setFileInputFiles(selectedFiles);
    renderSelectedFiles();
}

function moveFile(index, delta) {
    const newIndex = index + delta;
    if (newIndex < 0 || newIndex >= selectedFiles.length) return;

    const tmp = selectedFiles[index];
    selectedFiles[index] = selectedFiles[newIndex];
    selectedFiles[newIndex] = tmp;

    setFileInputFiles(selectedFiles);
    renderSelectedFiles();
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    setFileInputFiles(selectedFiles);
    renderSelectedFiles();
}

function renderSelectedFiles() {
    const list = document.getElementById('file-list');

    if (!selectedFiles.length) {
        list.classList.add('hidden');
        list.innerHTML = '';
        return;
    }

    list.classList.remove('hidden');

    list.innerHTML = selectedFiles.map((file, i) => {
        const type = isAudioFile(file.name) ? 'Аудио' : 'Видео';
        const subtitle = `${type} • ${formatFileSize(file.size)}`;
        const upDisabled = i === 0 ? 'disabled' : '';
        const downDisabled = i === selectedFiles.length - 1 ? 'disabled' : '';

        return `
            <div class="file-item">
                <div class="file-meta">
                    <div class="file-title">${escapeHtml(file.name)}</div>
                    <div class="file-subtitle">${escapeHtml(subtitle)}</div>
                </div>
                <div class="file-actions">
                    <button class="icon-btn" onclick="moveFile(${i}, -1)" ${upDisabled} title="Выше">↑</button>
                    <button class="icon-btn" onclick="moveFile(${i}, 1)" ${downDisabled} title="Ниже">↓</button>
                    <button class="icon-btn" onclick="removeFile(${i})" title="Убрать">×</button>
                </div>
            </div>
        `;
    }).join('');
}

function showProgress() {
    document.getElementById('progress-section').classList.remove('hidden');
}

function hideProgress() {
    document.getElementById('progress-section').classList.add('hidden');
}

function showResultsSummary(batchInfo) {
    document.getElementById('stat-success').textContent = batchInfo.completed_count || 0;
    document.getElementById('stat-errors').textContent = batchInfo.error_count || 0;
    document.getElementById('stat-total').textContent = batchInfo.total_count || 0;

    document.getElementById('results-section').classList.remove('hidden');
}

function hideResults() {
    document.getElementById('results-section').classList.add('hidden');
}

function showError(message) {
    document.getElementById('error-message').textContent = message;
    document.getElementById('error-section').classList.remove('hidden');

    if (progressCheckInterval) {
        clearInterval(progressCheckInterval);
        progressCheckInterval = null;
    }
}

function hideError() {
    document.getElementById('error-section').classList.add('hidden');
}

function resetProgress() {
    document.getElementById('progress-bar-fill').style.width = '0%';
    document.getElementById('progress-percentage').textContent = '0%';
    document.getElementById('current-stage-name').textContent = 'В очереди';
    document.getElementById('stage-description').textContent = 'Ожидание начала обработки';

    document.getElementById('batch-current-file').textContent = '—';
    document.getElementById('batch-next-file').textContent = '—';
    document.getElementById('batch-queue').innerHTML = '';
}

function statusToText(status) {
    const map = {
        queued: 'В очереди',
        created: 'В очереди',
        processing: 'В работе',
        completed: 'Готово',
        error: 'Ошибка'
    };
    return map[status] || status;
}

function renderBatchQueue(items, currentTaskId) {
    const container = document.getElementById('batch-queue');

    if (!Array.isArray(items) || !items.length) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = items.map(it => {
        const isProcessing = it.task_id === currentTaskId && it.status === 'processing';
        const cls = isProcessing ? 'batch-queue-item processing' : 'batch-queue-item';
        const name = escapeHtml(it.file_name || it.task_id);

        let statusText = statusToText(it.status);
        if (it.status === 'processing') {
            statusText = `${statusText} • ${it.percent || 0}%`;
        }

        return `
            <div class="${cls}">
                <div class="batch-queue-name">${name}</div>
                <div class="batch-queue-status">${escapeHtml(statusText)}</div>
            </div>
        `;
    }).join('');
}

function updateStageDisplay(stageDisplay) {
    if (!stageDisplay || typeof stageDisplay !== 'object') {
        return;
    }

    document.getElementById('current-stage-name').textContent = stageDisplay.display_name || 'Обработка';
    document.getElementById('stage-description').textContent = stageDisplay.description || '';
}

function handleBatchProgress(batchInfo) {
    const items = batchInfo.items || [];
    const currentTaskId = batchInfo.current_task_id;
    const nextTaskId = batchInfo.next_task_id;

    const currentItem = items.find(it => it.task_id === currentTaskId) || null;
    const nextItem = items.find(it => it.task_id === nextTaskId) || null;

    document.getElementById('batch-current-file').textContent = currentItem ? (currentItem.file_name || currentItem.task_id) : '—';
    document.getElementById('batch-next-file').textContent = nextItem ? (nextItem.file_name || nextItem.task_id) : '—';

    const percent = currentItem ? (currentItem.percent || 0) : 0;
    document.getElementById('progress-bar-fill').style.width = percent + '%';
    document.getElementById('progress-percentage').textContent = percent + '%';

    if (currentItem && currentItem.current_stage_display) {
        updateStageDisplay(currentItem.current_stage_display);
    }

    renderBatchQueue(items, currentTaskId);

    // Persist + render completed tasks
    items
        .filter(it => it.status === 'completed')
        .forEach(it => addCompletedTask({ task_id: it.task_id, file_name: it.file_name }));

    renderCompletedResults();

    const total = batchInfo.total_count || items.length;
    const finished = total > 0 && (batchInfo.completed_count + batchInfo.error_count) >= total;

    if (finished) {
        stopProgressTracking();
        hideProgress();
        showResultsSummary(batchInfo);
        document.getElementById('process-btn').disabled = false;
        localStorage.removeItem(STORAGE_KEYS.activeBatchId);
        currentBatchId = null;
    }
}

function stopProgressTracking() {
    if (progressCheckInterval) {
        clearInterval(progressCheckInterval);
        progressCheckInterval = null;
    }
}

function startProgressTracking() {
    stopProgressTracking();

    progressCheckInterval = setInterval(async () => {
        if (!currentBatchId) {
            stopProgressTracking();
            return;
        }

        try {
            const response = await fetch(`/batch/progress/${encodeURIComponent(currentBatchId)}?client_id=${encodeURIComponent(clientId)}`, {
                headers: { 'X-Client-Id': clientId }
            });

            if (!response.ok) {
                if (response.status === 404) {
                    stopProgressTracking();
                    localStorage.removeItem(STORAGE_KEYS.activeBatchId);
                    currentBatchId = null;
                    hideProgress();
                    document.getElementById('process-btn').disabled = false;
                    showError('Задача не найдена (возможно, сервер был перезапущен)');
                }
                return;
            }

            const data = await response.json();
            handleBatchProgress(data);

        } catch (error) {
            console.error('Ошибка при проверке прогресса:', error);
        }
    }, 2000);
}

async function startProcessing() {
    const urlInput = document.getElementById('url-input');
    const summaryType = document.getElementById('summary-type').value;
    const processBtn = document.getElementById('process-btn');

    hideError();
    hideResults();
    resetProgress();

    const isFileTabActive = document.getElementById('file-tab').classList.contains('active');

    const formData = new FormData();

    if (isFileTabActive) {
        if (!selectedFiles.length) {
            showError('Пожалуйста, выберите один или несколько файлов');
            return;
        }

        selectedFiles.forEach(file => formData.append('files', file, file.name));
    } else {
        const url = urlInput.value.trim();
        if (!url) {
            showError('Пожалуйста, укажите ссылку');
            return;
        }
        formData.append('url', url);
    }

    formData.append('summary_type', summaryType);

    showProgress();
    processBtn.disabled = true;

    try {
        const response = await fetch('/batch/process', {
            method: 'POST',
            headers: { 'X-Client-Id': clientId },
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Ошибка сервера');
        }

        const data = await response.json();
        currentBatchId = data.batch_id;
        localStorage.setItem(STORAGE_KEYS.activeBatchId, currentBatchId);

        startProgressTracking();

    } catch (error) {
        showError(error.message);
        hideProgress();
        processBtn.disabled = false;
    }
}

function resumeFromStorage() {
    const activeBatchId = localStorage.getItem(STORAGE_KEYS.activeBatchId);
    if (!activeBatchId) {
        return;
    }

    currentBatchId = activeBatchId;
    resetProgress();
    hideError();
    hideResults();
    showProgress();
    document.getElementById('process-btn').disabled = true;
    startProgressTracking();
}

document.addEventListener('DOMContentLoaded', function() {
    ensureClientId();
    renderCompletedResults();

    const fileInput = document.getElementById('file-input');
    const fileUploadZone = document.getElementById('file-upload-zone');

    fileInput.addEventListener('change', function() {
        onFilesSelected(this.files);
    });

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
        if (files && files.length > 0) {
            onFilesSelected(files);
        }
    });

    resumeFromStorage();
});

// Expose to global scope for inline handlers
window.switchTab = switchTab;
window.startProcessing = startProcessing;
window.moveFile = moveFile;
window.removeFile = removeFile;
window.clearHistory = clearHistory;
