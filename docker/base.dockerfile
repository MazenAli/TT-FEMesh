FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    ninja-build \
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN rm requirements.txt
