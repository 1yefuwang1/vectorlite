FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    ninja-build \
    g++ \
    python3 \
    curl \
    zip \
    unzip \
    git \
    pkg-config \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install cmake 
ENV CMAKE_VERSION=3.29.2

RUN curl -L "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh" -o cmake.sh \
    && chmod +x cmake.sh \
    && ./cmake.sh --skip-license --prefix=/usr/local \
    && rm cmake.sh

WORKDIR /workspace
