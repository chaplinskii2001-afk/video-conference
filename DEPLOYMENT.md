# Инструкция по развертыванию на сервере

## Предварительные требования

### 1. Установка Docker

```bash
# Обновить систему
sudo apt-get update
sudo apt-get upgrade -y

# Установить зависимости
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Добавить официальный GPG ключ Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Добавить репозиторий Docker
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Установить Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Проверить установку
docker --version
docker compose version
```

### 2. Установка Nvidia Driver

```bash
# Проверить наличие GPU
lspci | grep -i nvidia

# Установить драйвер (для Ubuntu)
sudo apt-get install -y nvidia-driver-525

# Перезагрузить систему
sudo reboot

# После перезагрузки проверить
nvidia-smi
```

### 3. Установка Nvidia Container Toolkit

```bash
# Настроить репозиторий
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Установить nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Настроить Docker для использования Nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Перезапустить Docker
sudo systemctl restart docker

# Проверить работу GPU в Docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Развертывание приложения

### 1. Клонирование репозитория

```bash
# Перейти в директорию проектов
cd /home/your-user/

# Клонировать репозиторий
git clone https://github.com/your-username/video-conference-processor.git
cd video-conference-processor
```

### 2. Настройка окружения

```bash
# Создать .env файл
cp .env.example .env

# Отредактировать .env и добавить HF_TOKEN
nano .env
# Вставить: HF_TOKEN=your_actual_token_here
```

### 3. Создание директорий

```bash
# Создать необходимые директории
mkdir -p logs uploads results

# Установить права доступа
chmod 755 logs uploads results
```

### 4. Сборка и запуск

```bash
# Собрать Docker образ (займет 20-40 минут)
docker compose build

# Запустить контейнер
docker compose up -d

# Проверить логи
docker compose logs -f
```

### 5. Проверка работы

```bash
# Проверить статус контейнера
docker compose ps

# Проверить здоровье приложения
curl http://localhost:8000/health

# Проверить GPU
curl http://localhost:8000/gpu-status
```

## Настройка автозапуска

### Systemd service (рекомендуется)

```bash
# Создать systemd service файл
sudo nano /etc/systemd/system/video-processor.service
```

Содержимое файла:

```ini
[Unit]
Description=Video Conference Processor
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/your-user/video-conference-processor
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=your-user

[Install]
WantedBy=multi-user.target
```

Активировать сервис:

```bash
# Перезагрузить systemd
sudo systemctl daemon-reload

# Включить автозапуск
sudo systemctl enable video-processor.service

# Запустить сервис
sudo systemctl start video-processor.service

# Проверить статус
sudo systemctl status video-processor.service
```

## Настройка Nginx (опционально)

Для использования с доменным именем и HTTPS:

```bash
# Установить Nginx
sudo apt-get install -y nginx

# Создать конфигурацию
sudo nano /etc/nginx/sites-available/video-processor
```

Содержимое:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 2G;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (если потребуется)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Таймауты для длительных запросов
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }
}
```

Активировать:

```bash
# Создать симлинк
sudo ln -s /etc/nginx/sites-available/video-processor /etc/nginx/sites-enabled/

# Проверить конфигурацию
sudo nginx -t

# Перезапустить Nginx
sudo systemctl restart nginx
```

### Установка SSL (Let's Encrypt)

```bash
# Установить Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Получить сертификат
sudo certbot --nginx -d your-domain.com

# Автообновление сертификата (уже настроено автоматически)
sudo certbot renew --dry-run
```

## Мониторинг и обслуживание

### Просмотр логов

```bash
# Логи Docker контейнера
docker compose logs -f

# Логи приложения
tail -f logs/app.log

# Логи Nginx
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Обновление приложения

```bash
# Перейти в директорию проекта
cd /home/your-user/video-conference-processor

# Получить последние изменения
git pull

# Пересобрать образ (если изменились зависимости)
docker compose build

# Перезапустить контейнер
docker compose down
docker compose up -d
```

### Очистка старых данных

```bash
# Очистить старые результаты (старше 7 дней)
find results/ -type f -mtime +7 -delete
find uploads/ -type f -mtime +7 -delete

# Очистить старые логи (старше 30 дней)
find logs/ -type f -mtime +30 -delete

# Очистить неиспользуемые Docker образы
docker system prune -a --volumes
```

### Резервное копирование

```bash
# Создать backup скрипт
nano backup.sh
```

Содержимое:

```bash
#!/bin/bash
BACKUP_DIR="/backup/video-processor"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup конфигурации
tar -czf $BACKUP_DIR/config_$DATE.tar.gz \
    .env docker-compose.yml

# Backup результатов (опционально)
tar -czf $BACKUP_DIR/results_$DATE.tar.gz results/

# Удалить старые backup (старше 30 дней)
find $BACKUP_DIR -type f -mtime +30 -delete
```

Сделать исполняемым и добавить в cron:

```bash
chmod +x backup.sh

# Добавить в crontab (каждый день в 2:00)
crontab -e
# Добавить строку:
# 0 2 * * * /home/your-user/video-conference-processor/backup.sh
```

## Troubleshooting

### Контейнер не запускается

```bash
# Проверить логи
docker compose logs

# Проверить статус GPU
nvidia-smi

# Перезапустить Docker
sudo systemctl restart docker
docker compose up -d
```

### GPU не обнаружена в контейнере

```bash
# Проверить nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Проверить GPU в контейнере
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Ошибка "CUDA out of memory"

```bash
# Проверить использование GPU
nvidia-smi

# Перезапустить контейнер
docker compose restart

# Если проблема сохраняется, проверить другие процессы
sudo fuser -v /dev/nvidia*
```

### Проблемы с сетью

```bash
# Проверить порты
sudo netstat -tulpn | grep 8000

# Проверить firewall
sudo ufw status
sudo ufw allow 8000/tcp  # если нужно
```

## Мониторинг производительности

### Установка monitoring tools

```bash
# Установить htop для мониторинга CPU/RAM
sudo apt-get install -y htop

# Установить nvtop для мониторинга GPU
sudo apt-get install -y nvtop

# Запустить мониторинг
htop
nvtop
```

### Prometheus + Grafana (опционально)

Для продвинутого мониторинга можно настроить Prometheus и Grafana, но это выходит за рамки базовой установки.

## Безопасность

### Firewall

```bash
# Установить UFW
sudo apt-get install -y ufw

# Разрешить SSH
sudo ufw allow 22/tcp

# Разрешить HTTP/HTTPS (если используется Nginx)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Если НЕ используется Nginx, разрешить порт приложения
# sudo ufw allow 8000/tcp

# Включить firewall
sudo ufw enable
```

### Обновления безопасности

```bash
# Настроить автоматические обновления безопасности
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```




