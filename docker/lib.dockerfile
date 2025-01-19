FROM ttfemesh-base:latest

WORKDIR /app

COPY requirements.txt ./
COPY ./ttfemesh ./ttfemesh
COPY README.md ./
COPY pyproject.toml ./
COPY setup.py ./

RUN pip install --no-cache-dir .

RUN rm pyproject.toml
RUN rm setup.py
RUN rm -rf ttfemesh*
RUN rm -rf build
RUN rm requirements.txt
