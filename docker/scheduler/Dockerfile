FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends cron && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY services/collector/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/collector/src/ /app/src/
ENV PYTHONPATH=/app/src
COPY docker/scheduler/crontab /etc/cron.d/algo-scheduler
RUN chmod 0644 /etc/cron.d/algo-scheduler && crontab /etc/cron.d/algo-scheduler
COPY docker/scheduler/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
