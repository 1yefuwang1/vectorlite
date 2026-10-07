FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    ninja-build g++ python3 curl zip unzip git pkg-config ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CMAKE_VERSION=3.29.2
RUN case "$(uname -m)" in \
        x86_64) cmake_arch=x86_64 ;; \
        aarch64) cmake_arch=aarch64 ;; \
        *) echo "Unsupported container architecture"; exit 1 ;; \
    esac \
    && curl --proto '=https' --tlsv1.2 -sSfL \
       "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${cmake_arch}.sh" \
       -o /tmp/cmake.sh \
    && sh /tmp/cmake.sh --skip-license --prefix=/usr/local \
    && rm /tmp/cmake.sh

ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable \
    && rustup component add clippy rustfmt

# uv-managed Python has loadable SQLite extension support for the SQL suites.
ENV UV_INSTALL_DIR=/usr/local/bin \
    UV_NO_MODIFY_PATH=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_BIN_DIR=/usr/local/bin
RUN curl --proto '=https' --tlsv1.2 -sSf https://astral.sh/uv/install.sh | sh \
    && uv python install 3.14 --default \
    && uv pip install --python /usr/local/bin/python --system pip pytest numpy

WORKDIR /workspace
