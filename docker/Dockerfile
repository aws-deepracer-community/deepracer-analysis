FROM ubuntu:24.04

ENV TZ="America/New_York"
ENV DEBIAN_FRONTEND="noninteractive"
# Keep the venv on PATH so all subsequent RUN/CMD steps use it
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

LABEL maintainer="AWS Deepracer Community" \
      description="Log Analysis for DeepRacer" \
      version=2.0

# System dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends \
        python3 \
        python3-venv \
        libsm6 \
        libxext6 \
        libxrender1 \
        git \
        wget \
        gosu \
        -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create virtual environment, install dependencies, then make it world-readable
RUN python3 -m venv "${VIRTUAL_ENV}"

WORKDIR /workspace
RUN mkdir -p /workspace/analysis && chown -R ubuntu:ubuntu /workspace

# Install Python dependencies into the venv (as root, venv is writable at build time)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    chmod -R a+rX "${VIRTUAL_ENV}"

# Copy notebook content
COPY --chown=ubuntu:ubuntu *.ipynb /workspace/
COPY --chown=ubuntu:ubuntu tracks /workspace/tracks
COPY --chown=ubuntu:ubuntu reward /workspace/reward
COPY --chown=ubuntu:ubuntu logs /workspace/logs

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Disable the "Would you like to get notified about Jupyter news?" popup
COPY docker/jupyter_overrides.json /opt/venv/share/jupyter/lab/settings/overrides.json

EXPOSE 8888
VOLUME ["/workspace/analysis", "/home/ubuntu/.aws"]
# Entrypoint runs as root, remaps UID/GID, then drops to notebook user
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--notebook-dir=/workspace"]
