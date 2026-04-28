FROM python:3.11-slim

WORKDIR /sandbox

RUN python -m pip install --upgrade pip && python -m pip install --default-timeout=300 --retries=10 --no-cache-dir pandas==2.2.3 polars==1.27.1 openpyxl==3.1.5

COPY runner.py /sandbox/runner.py

CMD ["python", "/sandbox/runner.py"]