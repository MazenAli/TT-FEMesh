FROM ttfemesh-dev:latest

RUN pip install --no-cache-dir notebook

WORKDIR /app/notebooks
