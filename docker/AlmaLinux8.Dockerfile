# This is used to generate the image with dependencies that CI.Dockerfile relies on.
# For instructions for deploying this, check docs/release-guide/push-docker-image.md.
FROM almalinux:8

RUN dnf upgrade -y --refresh
RUN dnf install -y 'dnf-command(config-manager)'

RUN dnf config-manager --add-repo=https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Extra repositories that have newer versions of some packages
RUN dnf install -y -q epel-release
RUN dnf config-manager --set-enabled powertools

RUN dnf install -y -q \
    git \
    git-lfs \
    wget \
    gcc-toolset-11 \
    gcc-toolset-11-libubsan-devel \
    make \
    doxygen \
    curl-devel \
    zlib-devel \
    perl-Data-Dumper \
    perl-Thread-Queue \
    perl-IPC-Cmd \
    wget \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    zlib-devel \
    sqlite-devel \
    xz-devel

# Install the nvidia driver.
RUN dnf module install -y -q nvidia-driver:535-dkms

# Enables gcc 11 for use within the docker image.
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /etc/bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Install newer version of Python
RUN wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz && \
    tar xzf Python-3.9.6.tgz && \
    cd Python-3.9.6 && \
    ./configure --enable-optimizations --enable-loadable-sqlite-extensions && \
    make altinstall && \
    cd .. && \
    rm Python-3.9.6.tgz && \
    rm -rf Python-3.9.6 && \
    ln -sf /usr/local/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3

# Install newer version of CMake
RUN wget https://cmake.org/files/v3.24/cmake-3.24.2.tar.gz && \
    tar xzf cmake-3.24.2.tar.gz && \
    cd cmake-3.24.2 && \
    ./bootstrap --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm cmake-3.24.2.tar.gz && \
    rm -rf cmake-3.24.2 && \
    ln -sf /usr/local/bin/cmake /usr/bin/cmake && \
    ln -sf /usr/local/bin/ctest /usr/bin/ctest && \
    ln -sf /usr/local/bin/cpack /usr/bin/cpack

RUN pip3 install conan==1.63.0 && \
    pip3 install gcovr

WORKDIR /var/app

ENTRYPOINT ["/bin/bash", "--login", "-c"]
