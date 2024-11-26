FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-git.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-git.txt

RUN rm requirements.txt requirements-git.txt
