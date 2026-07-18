FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ARG BUILD_VERSION="0.1.2"
ARG BUILD_ARCH="amd64"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENCV_FOR_THREADS_NUM=1 \
    FARMV_DATA_DIR=/data

LABEL io.hass.version="${BUILD_VERSION}" \
      io.hass.type="app" \
      io.hass.arch="${BUILD_ARCH}" \
      org.opencontainers.image.title="FarmBot Vision" \
      org.opencontainers.image.description="Lightweight FarmBot canopy analysis for Home Assistant" \
      org.opencontainers.image.source="https://github.com/sambiam/Weed-analyzer-for-Farmbot" \
      org.opencontainers.image.licenses="MIT"

RUN groupadd --system farmvision && useradd --system --gid farmvision --home /app farmvision
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt
COPY src ./src
COPY run.sh ./run.sh
RUN chmod 0755 ./run.sh && mkdir -p /data && chown -R farmvision:farmvision /app /data
USER farmvision
EXPOSE 8099
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8099/health', timeout=3)"
CMD ["./run.sh"]
