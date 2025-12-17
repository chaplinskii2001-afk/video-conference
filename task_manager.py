import uuid
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
import logging

from processing.stages import get_display_stage

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class TaskManager:
    def __init__(self):
        self.tasks_by_user: Dict[str, Dict[str, Dict]] = {}
        self.task_to_user: Dict[str, str] = {}

        self.batches_by_user: Dict[str, Dict[str, Dict]] = {}
        self.batch_to_user: Dict[str, str] = {}

        self.cleanup_interval = 300  # 5 минут
        self.storage_file = "results/tasks.json"
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        self.load_state()

    def _get_tomsk_time(self):
        """Получение текущего времени в Томске (UTC+7)."""
        tomsk_tz = timezone(timedelta(hours=7))
        return datetime.now(tomsk_tz)

    def save_state(self):
        """Сохраняет текущее состояние в файл."""
        try:
            state = {
                "tasks_by_user": self.tasks_by_user,
                "task_to_user": self.task_to_user,
                "batches_by_user": self.batches_by_user,
                "batch_to_user": self.batch_to_user,
            }
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(state, f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка при сохранении состояния задач: {e}")

    def load_state(self):
        """Загружает состояние из файла."""
        if not os.path.exists(self.storage_file):
            return

        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Восстанавливаем словари
            self.task_to_user = state.get("task_to_user", {})
            self.batch_to_user = state.get("batch_to_user", {})

            # Восстанавливаем tasks_by_user и конвертируем даты обратно
            self.tasks_by_user = state.get("tasks_by_user", {})
            for user_id, tasks in self.tasks_by_user.items():
                for task_id, task in tasks.items():
                    if "start_time" in task and task["start_time"]:
                        task["start_time"] = datetime.fromisoformat(task["start_time"])
                    if "end_time" in task and task["end_time"]:
                        task["end_time"] = datetime.fromisoformat(task["end_time"])

                    # Если задача была "в работе", помечаем как ошибку (сервер перезагрузился)
                    if task.get("status") == "processing":
                        task["status"] = "error"
                        task["error"] = "Сервер был перезагружен во время обработки"
                        task["logs"].append(
                            f"[{self._get_tomsk_time().strftime('%H:%M:%S')}] Ошибка: Сервер был перезагружен"
                        )
                        logger.warning(f"Задача {task_id} помечена как прерванная из-за перезагрузки")

            # Восстанавливаем batches_by_user и конвертируем даты
            self.batches_by_user = state.get("batches_by_user", {})
            for user_id, batches in self.batches_by_user.items():
                for batch_id, batch in batches.items():
                    if "created_at" in batch and batch["created_at"]:
                        batch["created_at"] = datetime.fromisoformat(batch["created_at"])

            logger.info("Состояние задач успешно восстановлено")
        except Exception as e:
            logger.error(f"Ошибка при загрузке состояния задач: {e}")
            # В случае ошибки начинаем с чистого листа
            self.tasks_by_user = {}
            self.task_to_user = {}
            self.batches_by_user = {}
            self.batch_to_user = {}

    def _ensure_user(self, user_id: str):
        if user_id not in self.tasks_by_user:
            self.tasks_by_user[user_id] = {}
        if user_id not in self.batches_by_user:
            self.batches_by_user[user_id] = {}

    def create_task(
        self,
        *,
        user_id: str,
        task_id: str = None,
        file_name: str = None,
        source_type: str = "file",
        summary_type: str = "standard",
        batch_id: str = None,
        order_index: int = None,
        total_in_batch: int = None,
    ) -> str:
        """Создает новую задачу и возвращает её ID."""
        if task_id is None:
            task_id = str(uuid.uuid4())

        self._ensure_user(user_id)

        self.tasks_by_user[user_id][task_id] = {
            "task_id": task_id,
            "user_id": user_id,
            "status": "queued",  # queued -> processing -> completed -> error
            "percent": 0,
            "current_stage": "queued",
            "current_stage_display": get_display_stage("queued"),
            "logs": [],
            "start_time": self._get_tomsk_time(),
            "result": None,
            "error": None,
            "file_name": file_name,
            "source_type": source_type,
            "summary_type": summary_type,
            "batch_id": batch_id,
            "order_index": order_index,
            "total_in_batch": total_in_batch,
        }

        self.task_to_user[task_id] = user_id
        logger.info(f"Создана новая задача: {task_id} (user_id={user_id})")
        self.save_state()
        return task_id

    def create_batch(
        self,
        *,
        user_id: str,
        items: List[Dict],
        summary_type: str = "standard",
    ) -> Tuple[str, List[Dict]]:
        """Создает батч и задачи для каждого элемента в указанном порядке."""
        self._ensure_user(user_id)

        batch_id = str(uuid.uuid4())
        created_at = self._get_tomsk_time()

        batch_items: List[Dict] = []
        total = len(items)

        for index, item in enumerate(items):
            task_id = self.create_task(
                user_id=user_id,
                file_name=item.get("file_name"),
                source_type=item.get("source_type", "file"),
                summary_type=summary_type,
                batch_id=batch_id,
                order_index=index,
                total_in_batch=total,
            )
            batch_items.append(
                {
                    "task_id": task_id,
                    "file_name": item.get("file_name"),
                    "source_type": item.get("source_type", "file"),
                    "order_index": index,
                }
            )

        self.batches_by_user[user_id][batch_id] = {
            "batch_id": batch_id,
            "user_id": user_id,
            "created_at": created_at,
            "summary_type": summary_type,
            "items": batch_items,
        }
        self.batch_to_user[batch_id] = user_id

        logger.info(
            f"Создан батч {batch_id} (user_id={user_id}, items={len(batch_items)})"
        )
        self.save_state()
        return batch_id, batch_items

    def task_belongs_to_user(self, *, user_id: str, task_id: str) -> bool:
        return self.task_to_user.get(task_id) == user_id

    def batch_belongs_to_user(self, *, user_id: str, batch_id: str) -> bool:
        return self.batch_to_user.get(batch_id) == user_id

    def _get_task_ref(self, task_id: str) -> Optional[Dict]:
        user_id = self.task_to_user.get(task_id)
        if not user_id:
            return None
        return self.tasks_by_user.get(user_id, {}).get(task_id)

    def update_progress(
        self, task_id: str, percent: int, stage: str, log_message: str = None
    ):
        """Обновляет прогресс задачи."""
        task = self._get_task_ref(task_id)
        if not task:
            logger.warning(f"Попытка обновить несуществующую задачу: {task_id}")
            return

        task["percent"] = percent
        task["current_stage"] = stage
        task["current_stage_display"] = get_display_stage(stage)
        task["status"] = "processing"

        if log_message:
            timestamp = self._get_tomsk_time().strftime("%H:%M:%S")
            task["logs"].append(f"[{timestamp}] {log_message}")

            if len(task["logs"]) > 50:
                task["logs"] = task["logs"][-50:]

        logger.debug(
            f"Обновлен прогресс задачи {task_id}: {percent}%, этап: {stage}"
        )
        self.save_state()

    def complete_task(self, task_id: str, result: Dict):
        """Отмечает задачу как завершенную."""
        task = self._get_task_ref(task_id)
        if not task:
            logger.warning(f"Попытка завершить несуществующую задачу: {task_id}")
            return

        task.update(
            {
                "status": "completed",
                "percent": 100,
                "current_stage": "completed",
                "current_stage_display": get_display_stage("completed"),
                "result": result,
                "end_time": self._get_tomsk_time(),
            }
        )

        timestamp = self._get_tomsk_time().strftime("%H:%M:%S")
        task["logs"].append(f"[{timestamp}] Обработка успешно завершена")

        logger.info(f"Задача {task_id} завершена успешно")
        self.save_state()

    def fail_task(self, task_id: str, error_message: str):
        """Отмечает задачу как завершенную с ошибкой."""
        task = self._get_task_ref(task_id)
        if not task:
            logger.warning(
                f"Попытка завершить с ошибкой несуществующую задачу: {task_id}"
            )
            return

        task.update(
            {
                "status": "error",
                "error": error_message,
                "end_time": self._get_tomsk_time(),
            }
        )

        timestamp = self._get_tomsk_time().strftime("%H:%M:%S")
        task["logs"].append(f"[{timestamp}] Ошибка: {error_message}")

        logger.error(f"Задача {task_id} завершена с ошибкой: {error_message}")
        self.save_state()

    def get_task_info(self, task_id: str) -> Optional[Dict]:
        """Возвращает информацию о задаче (без проверки прав)."""
        task = self._get_task_ref(task_id)
        if not task:
            return None
        return dict(task)

    def get_task_info_for_user(self, *, user_id: str, task_id: str) -> Optional[Dict]:
        if not self.task_belongs_to_user(user_id=user_id, task_id=task_id):
            return None
        return self.get_task_info(task_id)

    def get_batch_info_for_user(self, *, user_id: str, batch_id: str) -> Optional[Dict]:
        if not self.batch_belongs_to_user(user_id=user_id, batch_id=batch_id):
            return None

        batch = self.batches_by_user.get(user_id, {}).get(batch_id)
        if not batch:
            return None

        items = []
        for item in batch.get("items", []):
            task_id = item["task_id"]
            task = self.tasks_by_user.get(user_id, {}).get(task_id)
            if not task:
                continue

            items.append(
                {
                    "task_id": task_id,
                    "file_name": item.get("file_name") or task.get("file_name"),
                    "source_type": item.get("source_type") or task.get("source_type"),
                    "order_index": item.get("order_index"),
                    "status": task.get("status"),
                    "percent": task.get("percent"),
                    "current_stage": task.get("current_stage"),
                    "current_stage_display": task.get("current_stage_display"),
                    "result": task.get("result"),
                    "error": task.get("error"),
                }
            )

        items.sort(key=lambda x: x.get("order_index", 0))

        current_item = next(
            (it for it in items if it.get("status") == "processing"), None
        )
        if not current_item:
            current_item = next(
                (it for it in items if it.get("status") in {"queued", "created"}),
                None,
            )

        next_item = None
        if current_item:
            idx = current_item.get("order_index")
            next_item = next((it for it in items if it.get("order_index") == idx + 1), None)

        completed_count = len([it for it in items if it.get("status") == "completed"])
        error_count = len([it for it in items if it.get("status") == "error"])

        return {
            "batch_id": batch_id,
            "summary_type": batch.get("summary_type"),
            "created_at": batch.get("created_at"),
            "items": items,
            "current_task_id": current_item.get("task_id") if current_item else None,
            "next_task_id": next_item.get("task_id") if next_item else None,
            "completed_count": completed_count,
            "error_count": error_count,
            "total_count": len(items),
        }

    def cleanup_old_tasks(self):
        """Очищает старые задачи (старше 1 часа после завершения)."""
        now = self._get_tomsk_time()

        tasks_to_remove: List[Tuple[str, str]] = []

        for user_id, tasks in self.tasks_by_user.items():
            for task_id, task_info in tasks.items():
                end_time = task_info.get("end_time")
                if not end_time:
                    continue

                if now - end_time > timedelta(hours=1):
                    tasks_to_remove.append((user_id, task_id))

        for user_id, task_id in tasks_to_remove:
            self.tasks_by_user.get(user_id, {}).pop(task_id, None)
            self.task_to_user.pop(task_id, None)
            logger.info(f"Удалена старая задача: {task_id} (user_id={user_id})")

        batches_to_remove: List[Tuple[str, str]] = []
        for user_id, batches in self.batches_by_user.items():
            for batch_id, batch in batches.items():
                item_task_ids = [it.get("task_id") for it in batch.get("items", [])]
                if any(task_id in self.tasks_by_user.get(user_id, {}) for task_id in item_task_ids):
                    continue
                batches_to_remove.append((user_id, batch_id))

        for user_id, batch_id in batches_to_remove:
            self.batches_by_user.get(user_id, {}).pop(batch_id, None)
            self.batch_to_user.pop(batch_id, None)
            logger.info(f"Удален старый батч: {batch_id} (user_id={user_id})")
        
        self.save_state()


# Глобальный экземпляр менеджера задач
task_manager = TaskManager()
