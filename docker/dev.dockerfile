FROM tnfemesh-base:latest

WORKDIR /app

COPY requirements.txt requirements-git.txt requirements-dev.txt ./
COPY tnfemesh ./
COPY README.md ./
COPY pyproject.toml ./
COPY setup.py ./

RUN pip install --no-cache-dir -r requirements-dev.txt
RUN pip install --no-cache-dir -e .
