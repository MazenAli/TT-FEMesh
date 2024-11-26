FROM tnfemesh:latest

RUN pip install --no-cache-dir notebook

WORKDIR /app/notebooks
