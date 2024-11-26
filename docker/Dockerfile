FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY docker/requirements/requirements-pip.txt docker/requirements/requirements-git.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-pip.txt
RUN pip install --no-cache-dir -r requirements-git.txt
RUN rm requirements-pip.txt requirements-git.txt

COPY src/ ./src
COPY setup.py ./
COPY README.md ./
RUN pip install --no-cache-dir .
RUN rm setup.py
RUN rm -rf src
RUN rm -rf build
