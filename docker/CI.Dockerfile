FROM cesiumgs/omniverse-almalinux8-build:2023-11-02

WORKDIR /var/app

RUN git config --global --add safe.directory '*'

ENTRYPOINT ["/bin/bash", "--login"]
