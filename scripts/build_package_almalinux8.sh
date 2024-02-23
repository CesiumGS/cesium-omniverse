#!/bin/bash

# Make sure to run this script from the root cesium-omniverse directory

# Delete existing build folder
sudo rm -rf build-package

# Start the docker container
docker build --tag cesiumgs/cesium-omniverse:almalinux8 -f docker/AlmaLinux8.Dockerfile .
container_id=$(docker run -it -d --volume $PWD:/var/app cesiumgs/cesium-omniverse:almalinux8)

# Run package commands inside docker container
package_cmd="
cmake -B build-package -D CMAKE_BUILD_TYPE=Release -D CESIUM_OMNI_ENABLE_TESTS=OFF -D CESIUM_OMNI_ENABLE_DOCUMENTATION=OFF -D CESIUM_OMNI_ENABLE_SANITIZERS=OFF -D CESIUM_OMNI_ENABLE_LINTERS=OFF &&
cmake --build build-package --parallel 8 &&
cmake --build build-package --target install &&
cmake --build build-package --target package
"

docker exec ${container_id} /bin/sh -c "${package_cmd}"

# Clean up
docker stop ${container_id}
docker rm ${container_id}
