FROM ttfemesh-dev:latest

WORKDIR /app

COPY requirements-docs.txt ./
RUN pip install --no-cache-dir -r requirements-docs.txt
