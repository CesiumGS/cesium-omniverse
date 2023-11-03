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
    python39 \
    wget \
    gcc-toolset-11 \
    gcc-toolset-11-libubsan-devel \
    make \
    doxygen \
    curl-devel \
    zlib-devel \
    perl-Data-Dumper \
    perl-Thread-Queue \
    wget \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    zlib-devel \
    sqlite-devel \
    xz-devel

# Install the nvidia driver.
RUN dnf module install -y -q nvidia-driver:535-dkms

# /bin/gcc, /bin/gcov, /bin/ranlib and /bin/ar are old versions and not
# symbolic links, which prevents alternatives from working properly, so
# rename them
RUN mv /bin/gcc /bin/gcc-8 && \
    mv /bin/gcov /bin/gcov-8

# Create links to some of the custom packages
RUN alternatives --install /usr/bin/gcc gcc /opt/rh/gcc-toolset-11/root/usr/bin/gcc 100 && \
    alternatives --install /usr/bin/gcov gcov /opt/rh/gcc-toolset-11/root/usr/bin/gcov 100 && \
    alternatives --install /usr/bin/g++ g++ /opt/rh/gcc-toolset-11/root/usr/bin/g++ 100 && \
    alternatives --install /usr/bin/cc cc /opt/rh/gcc-toolset-11/root/usr/bin/gcc 100 && \
    alternatives --install /usr/bin/c++ c++ /opt/rh/gcc-toolset-11/root/usr/bin/g++ 100

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

RUN pip3 install conan==1.58.0 && \
    pip3 install gcovr

WORKDIR /var/app

ENTRYPOINT ["/bin/bash"]
