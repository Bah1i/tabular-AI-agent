FROM docker:26-cli AS docker-cli

FROM python:3.11-slim

WORKDIR /app

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --default-timeout=300 --retries=10 --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
