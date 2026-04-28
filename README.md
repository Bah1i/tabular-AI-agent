# Tabular AI Agent Full MVP

Полный MVP для магистерской: FastAPI + PostgreSQL + Redis/RQ + worker + DeepSeek + Docker sandbox + Langfuse + UI + метрики.

## Запуск

```bash
cp .env.example .env
docker build -t tabular-agent-sandbox:latest ./sandbox
docker compose up --build
```

UI: http://localhost:8000

