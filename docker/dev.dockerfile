FROM tnfemesh:latest

WORKDIR /app

COPY docker/requirements/requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
RUN rm requirements-dev.txt

COPY src/ ./src
COPY README.md ./
COPY setup.py ./
RUN pip install --no-cache-dir -e .
