FROM python:3.13-slim

RUN useradd --create-home --uid 10001 borealis
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .
USER borealis
WORKDIR /workspace
ENTRYPOINT ["borealis"]
CMD ["--help"]
