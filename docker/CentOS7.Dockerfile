# This is used to generate the image with dependencies that CI.Dockerfile relies on.
# For instructions for deploying this, check docs/release-guide/push-docker-image.md.
FROM centos:7

RUN yum update -y -q

# Add nvidia repository
RUN yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo

# Extra repositories that have newer versions of some packages
RUN yum install -y -q \
    epel-release \
    centos-release-scl

RUN yum install -y -q \
    # for gcc-11
    devtoolset-11-toolchain \
    # for ubsan
    devtoolset-11-libubsan-devel \
    # get a newer version of git and git-lfs
    # CentOS 7 comes with version 1.8.3.1 by default, which is from 2013!
    rh-git218 \
    rh-git218-git-lfs \
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
    xz-devel \
    nvidia-driver-branch-535.x86_64

# Create links to some of the custom packages
RUN update-alternatives --install /usr/bin/gcc gcc /opt/rh/devtoolset-11/root/usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/g++ g++ /opt/rh/devtoolset-11/root/usr/bin/g++ 100 && \
    update-alternatives --install /usr/bin/cc cc /opt/rh/devtoolset-11/root/usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/c++ c++ /opt/rh/devtoolset-11/root/usr/bin/g++ 100 && \
    update-alternatives --install /usr/bin/gcov gcov /opt/rh/devtoolset-11/root/usr/bin/gcov 100 && \
    update-alternatives --install /usr/bin/git git /opt/rh/rh-git218/root/usr/bin/git 100 && \
    update-alternatives --install /usr/bin/git-lfs git-lfs /opt/rh/rh-git218/root/usr/bin/git-lfs 100

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

ENTRYPOINT ["/bin/bash"]
