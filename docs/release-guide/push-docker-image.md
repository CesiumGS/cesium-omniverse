# Pushing the Docker Image for AlmaLinux 8 builds.

We use a docker image for our AlmaLinux 8 builds that contains all of our build dependencies, so we don't have to build the image from scratch on each build. This document outlines how to build and push this to Docker Hub.

## Installing Docker

Install [Docker Desktop](https://docs.docker.com/desktop/install/ubuntu/). You will need a license for this and access to our account.

On Linux, docker is run as root. To avoid the requirement for `sudo`, you should add your user to the `docker` group:

```shell
sudo usermod -aG docker $USER
```

To use the new group membership without logging out of your session
completely, you can "relogin" in the same shell by typing:
```shell
su - $USER
```

Note: this creates a new login shell and may behave differently from
your expectations in a windowed environment e.g., GNOME. In
particular, `ssh` logins and `git` may not work anymore.

## Building the container

Confirm that you have push access to the [container repo](https://hub.docker.com/r/cesiumgs/omniverse-almalinux8-build).

### Log in

Log into docker using:

```shell
docker login
```

### Build the docker image

After making your changes to the docker file, execute:

```shell
docker build --tag cesiumgs/omniverse-almalinux8-build:$TAGNAME -f docker/AlmaLinux8.Dockerfile . --no-cache
```

You should replace `TAGNAME` with the current date in `YYYY-MM-DD` format. So if it's the 29th of August, 2023, you would use `2023-08-29`.

### Push the image to Docker Hub

The build will take some time. Once it is finished, execute the following to push the image to Docker Hub:

```shell
docker push cesiumgs/omniverse-almalinux8-build:$TAGNAME
```

Again, you should replace `$TAGNAME` with the current date in `YYYY-MM-DD` format. So if it's the 29th of August, 2023, you would use `2023-08-29`.

### Update CI.Dockerfile

The `docker/CI.Dockerfile` file is used as part of the AlmaLinux 8 build step in our GitHub actions. You will need to update the version of the Docker image used to the tagged version you just uploaded.
