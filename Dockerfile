# Dockerfile to support portable execution of the ABM benchmarks, including FLAME GPU 2, hence using the CUDA base image.

# Must use an nvidia/cuda base image for libcuda.so redistribution
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

# Only support building for x86_64 CPU arch, as arch specific precompiled binaries are downloaded
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        'amd64') \
            # do nothing
            ;; \
        *) \
            echo >&2 "error: current arch (${arch}) not supported by this dockerfile." \
            exit 1; \
        ;; \
    esac;

# Create a new non-root user, to allow some commands to not be executed as root
RUN groupadd -r bench && useradd --shell /bin/bash --create-home --no-log-init -r -g bench bench
ENV DEBIAN_FRONTEND=noninteractive
# Update apt, install general / common dependencies.
# Doing this once makes it a little harder to split some aspects, but should improve build time by not attempting to reinstall python3 multiple times
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        parallel \
        python3 \
        python3-pip \
        python3-venv \
    ; \
    rm -rf /var/lib/apt/lists/*

# Install Dependencies for FLAME GPU 2 (console, no python)
# There are not yet binary releases of the c++ static library which can be installed independently of models.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        wget \
        tar \
    ; \
    rm -rf /var/lib/apt/lists/*; \
    # Remove old CMake
    apt-get remove -y cmake; \
    # Install newer CMake
    wget -qO - https://cmake.org/files/v3.22/cmake-3.22.6-linux-x86_64.tar.gz | tar -xzf - -C /opt; \
    ln -sf /opt/cmake-3.22.6-linux-x86_64/bin/cmake /usr/local/bin/cmake; \
    cmake --version; \
    gcc --version; \
    nvcc --version

# Install Dependencies for Agents.jl
ENV JULIA_VERSION_FULL 1.8.2
ENV JULIA_VERSION_MM 1.8
ENV JULIA_PATH /opt/julia
ENV PATH $JULIA_PATH/bin:$PATH
# ENV JULIA_DEPOT_PATH "/opt/julia-depot" # Julia depot needs to be user writable.
RUN set -eux; \
    curl -fL -o julia.tar.gz "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION_MM}/julia-${JULIA_VERSION_FULL}-linux-x86_64.tar.gz"; \
    mkdir -p ${JULIA_PATH}; \
    tar -xzf julia.tar.gz -C ${JULIA_PATH} --strip-components 1; \
    rm -rf julia.tar.gz; \
    julia --version

# Install Dependencies for NetLogo
ENV NETLOGO_VERSION_FULL 6.3.0
ENV NETLOGO_PATH /opt/netlogo
ENV PATH $NETLOGO_PATH/bin:$PATH
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        default-jre \
    ; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fL -o netlogo.tgz "https://ccl.northwestern.edu/netlogo/${NETLOGO_VERSION_FULL}/NetLogo-${NETLOGO_VERSION_FULL}-64.tgz"; \
    mkdir -p ${NETLOGO_PATH}; \
    tar -xzf netlogo.tgz -C ${NETLOGO_PATH} --strip-components 1; \
    rm -rf netlogo.tgz; \
    JAVA_HOME=/usr ${NETLOGO_PATH}/netlogo-headless.sh --version

# Install Mesa python package packages ependencies for Mesa
ENV MESA_VERSION 1.0
RUN set -eux; \
    python3 -m pip install mesa==${MESA_VERSION}; \
    python3 -c "import mesa; print(mesa.__version__)"

# Install Julia packages from project.toml into the container
COPY Project.toml .
# Install Julia packages as required by Project.toml?
RUN set -eux; \
    pwd; \
    ls; \
    julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate();'

# Copy the benchmark scripts / sources into /opt/ABM_Framework_Comparisons
COPY . /opt/ABM_Framework_Comparisons

# Configure and build flamegpu binaries into /opt/ABM_Framework_Comparisons
WORKDIR /opt/ABM_Framework_Comparisons

RUN cmake -S FLAMEGPU2 -B FLAMEGPU2/build .. -DCMAKE_CUDA_ARCHITECTURES="70;80" -DFLAMEGPU_SEATBELTS=OFF && cmake --build FLAMEGPU2/build --target all -j `nproc` && rm -rf FLAMEGPU2/build/CMakeCache.txt FLAMEGPU2/build/CMakeFiles FLAMEGPU2/build/Makefile FLAMEGPU2/build/_deps FLAMEGPU2/build/compile_commands.json