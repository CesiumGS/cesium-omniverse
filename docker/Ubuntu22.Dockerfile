# This is used to generate the image with dependencies that CI.Dockerfile relies on.
# For instructions for deploying this, check docs/release-guide/push-docker-image.md.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Update system
RUN apt-get update && \
    apt-get upgrade -y

# Get access to GCC 13
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update

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
    cmake \
    autoconf \
    autoconf-archive \
    automake \
    libtool \
    gcc-13 \
    g++-13

# Use gcc-13 as default compiler
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 100

# Allow mounted repos owned by host user
RUN git config --global --add safe.directory /var/app

# Initialize git-lfs
RUN git lfs install

# Python packages
RUN pip3 install --no-cache-dir \
    gcovr pybind11

# Install vcpkg
WORKDIR /home
RUN git clone https://github.com/microsoft/vcpkg.git
COPY vcpkg.json /tmp/vcpkg.json

WORKDIR /home/vcpkg
RUN VCPKG_BASELINE="$(python3 -c 'import json; print(json.load(open("/tmp/vcpkg.json"))["builtin-baseline"])')" && \
    git fetch origin "${VCPKG_BASELINE}" && \
    git checkout "${VCPKG_BASELINE}" && \
    ./bootstrap-vcpkg.sh

# Set environment variables
ENV VCPKG_ROOT="/home/vcpkg"
ENV PATH="${VCPKG_ROOT}:${PATH}"

RUN echo 'export PATH=$PATH:/home/vcpkg' >> ~/.bashrc && \
    echo 'export VCPKG_ROOT=/home/vcpkg' >> ~/.bashrc

WORKDIR /var/app

ENTRYPOINT ["/bin/bash", "--login"]
