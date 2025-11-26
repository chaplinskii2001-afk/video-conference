import asyncio
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.cleanup_interval = 300  # 5 минут

    def _get_tomsk_time(self):
        """Получение текущего времени в Томске (UTC+7)"""
        tomsk_tz = timezone(timedelta(hours=7))
        return datetime.now(tomsk_tz)

    def create_task(self, task_id: str = None) -> str:
        """Создает новую задачу и возвращает её ID"""
        if task_id is None:
            task_id = str(uuid.uuid4())

        self.tasks[task_id] = {
            'status': 'created',  # created -> processing -> completed -> error
            'percent': 0,
            'current_stage': 'waiting',
            'logs': [],
            'start_time': self._get_tomsk_time(),
            'result': None,
            'error': None
        }
        logger.info(f"Создана новая задача: {task_id}")
        return task_id

    def update_progress(self, task_id: str, percent: int, stage: str, log_message: str = None):
        """Обновляет прогресс задачи"""
        if task_id not in self.tasks:
            logger.warning(f"Попытка обновить несуществующую задачу: {task_id}")
            return

        self.tasks[task_id]['percent'] = percent
        self.tasks[task_id]['current_stage'] = stage
        self.tasks[task_id]['status'] = 'processing'

        if log_message:
            timestamp = self._get_tomsk_time().strftime('%H:%M:%S')
            log_entry = f"[{timestamp}] {log_message}"
            self.tasks[task_id]['logs'].append(log_entry)

            # Ограничиваем количество логов
            if len(self.tasks[task_id]['logs']) > 50:
                self.tasks[task_id]['logs'] = self.tasks[task_id]['logs'][-50:]

        logger.debug(f"Обновлен прогресс задачи {task_id}: {percent}%, этап: {stage}")

    def complete_task(self, task_id: str, result: Dict):
        """Отмечает задачу как завершенную"""
        if task_id not in self.tasks:
            logger.warning(f"Попытка завершить несуществующую задачу: {task_id}")
            return

        self.tasks[task_id].update({
            'status': 'completed',
            'percent': 100,
            'current_stage': 'completed',
            'result': result,
            'end_time': self._get_tomsk_time()
        })

        # Добавляем финальный лог
        timestamp = self._get_tomsk_time().strftime('%H:%M:%S')
        self.tasks[task_id]['logs'].append(f"[{timestamp}] Обработка успешно завершена")

        logger.info(f"Задача {task_id} завершена успешно")

    def fail_task(self, task_id: str, error_message: str):
        """Отмечает задачу как завершенную с ошибкой"""
        if task_id not in self.tasks:
            logger.warning(f"Попытка завершить с ошибкой несуществующую задачу: {task_id}")
            return

        self.tasks[task_id].update({
            'status': 'error',
            'error': error_message,
            'end_time': self._get_tomsk_time()
        })

        # Добавляем лог об ошибке
        timestamp = self._get_tomsk_time().strftime('%H:%M:%S')
        self.tasks[task_id]['logs'].append(f"[{timestamp}] Ошибка: {error_message}")

        logger.error(f"Задача {task_id} завершена с ошибкой: {error_message}")

    def get_task_info(self, task_id: str) -> Optional[Dict]:
        """Возвращает информацию о задаче"""
        return self.tasks.get(task_id)

    def cleanup_old_tasks(self):
        """Очищает старые задачи (старше 1 часа)"""
        now = self._get_tomsk_time()
        old_tasks = []

        for task_id, task_info in self.tasks.items():
            if 'end_time' in task_info:
                if now - task_info['end_time'] > timedelta(hours=1):
                    old_tasks.append(task_id)

        for task_id in old_tasks:
            del self.tasks[task_id]
            logger.info(f"Удалена старая задача: {task_id}")

# Глобальный экземпляр менеджера задач
task_manager = TaskManager()