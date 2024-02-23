# This is used to generate the image with dependencies that CI.Dockerfile relies on.
# For instructions for deploying this, check docs/release-guide/push-docker-image.md.
FROM almalinux:8

RUN dnf upgrade -y --refresh
RUN dnf install -y 'dnf-command(config-manager)'

# Extra repositories that have newer versions of some packages
RUN dnf install -y -q epel-release
RUN dnf config-manager --set-enabled powertools
RUN dnf config-manager --add-repo=https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

RUN dnf install -y -q \
    git \
    git-lfs \
    python39 \
    gcc-toolset-11 \
    make \
    doxygen \
    perl

# Install the nvidia driver.
RUN dnf module install -y -q nvidia-driver:535-dkms

# /bin/gcc, /bin/gcov, /bin/ranlib and /bin/ar are old versions and not
# symbolic links, which prevents update-alternatives from working properly, so
# rename them
RUN mv /bin/gcc /bin/gcc-8 && \
    mv /bin/gcov /bin/gcov-8 && \
    mv /bin/ranlib /bin/ranlib-2.30 && \
    mv /bin/ar /bin/ar-2.30

# Create links to some of the custom packages
RUN update-alternatives --install /usr/bin/gcc gcc /opt/rh/gcc-toolset-11/root/usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/gcov gcov /opt/rh/gcc-toolset-11/root/usr/bin/gcov 100 && \
    update-alternatives --install /usr/bin/g++ g++ /opt/rh/gcc-toolset-11/root/usr/bin/g++ 100 && \
    update-alternatives --install /usr/bin/cc cc /opt/rh/gcc-toolset-11/root/usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/c++ c++ /opt/rh/gcc-toolset-11/root/usr/bin/g++ 100 && \
    update-alternatives --install /usr/bin/ar ar /opt/rh/gcc-toolset-11/root/usr/bin/ar 100 && \
    update-alternatives --install /usr/bin/ranlib ranlib /opt/rh/gcc-toolset-11/root/usr/bin/ranlib 100

# Alma8's default version of CMake is 3.20.2, which is lower than project's min
# pip has a much newer version of CMake
RUN pip3 install conan==1.60.0 && \
    pip3 install gcovr && \
    pip3 install cmake

# Prevent git error when building within docker container
RUN git config --global --add safe.directory /var/app

WORKDIR /var/app

ENTRYPOINT ["/bin/bash", "--login"]
