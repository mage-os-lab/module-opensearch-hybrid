FROM python:3.13.7-alpine3.22@sha256:9ba6d8cbebf0fb6546ae71f2a1c14f6ffd2fdab83af7fa5669734ef30ad48844

ENV ENCODER_HOST=0.0.0.0 \
    ENCODER_PORT=8080 \
    PYTHONPATH=/srv/encoder/src \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv/encoder
COPY src /srv/encoder/src

USER 65532:65532
EXPOSE 8080

HEALTHCHECK --interval=5s --timeout=2s --retries=6 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/ready', timeout=1).read(1)"]
ENTRYPOINT ["python", "-m", "mageos_opensearch_hybrid_encoder.server"]
