FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# состояние (SQLite, кэши file_id, разметку картин) держим в volume
ENV BOT_DATA=/data

ENTRYPOINT ["/entrypoint.sh"]
