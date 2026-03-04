# This is used to generate the image with dependencies that CI.Dockerfile relies on.
# For instructions for deploying this, check docs/release-guide/push-docker-image.md.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Update system
RUN apt-get update && \
    apt-get upgrade -y

# Core build dependencies
RUN apt-get install -y \
    git \
    git-lfs \
    wget \
    build-essential \
    make \
    doxygen \
    curl \
    libcurl4-openssl-dev \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libffi-dev \
    libsqlite3-dev \
    liblzma-dev \
    zip \
    unzip \
    perl \
    pkg-config \
    python3 \
    python3-pip \
    cmake

# Allow mounted repos owned by host user
RUN git config --global --add safe.directory /var/app

# Initialize git-lfs
RUN git lfs install

# Python packages
RUN pip3 install --no-cache-dir \
    conan==1.63.0 \
    gcovr

WORKDIR /var/app

ENTRYPOINT ["/bin/bash", "--login"]
