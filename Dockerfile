# Dockerfile to support portable execution of the ABM benchmarks, including FLAME GPU 2, hence using the CUDA base image.

# @todo - use a multi-stage build to compile the benchmarks into a final image which can simply be invoked to run the full set of benchmarks. This can use a non-devel cuda base image to shrink the file size. 
# @todo - gpg & sha validation of remote downloads.
# @todo - flag the secitosn of this which are arch specific, i.e. precompiled binaries may be linux/amd64 only.
# @todo - install specific versions of packages as required (e.g. mesa). It is not clear which versions these are currently. I.e. add a requiremetns.txt for mesa. At the very least document it.

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# For now, abort for non-x86 arch's just incase, as uri's are baked for x86
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        'samd64') \
            # do nothing
            ;; \
        *) \
            echo >&2 "error: current arch (${arch}) not supperted by this dockerfile.;" \
            exit 1; \
        ;; \
    esac;

# Update apt, install some general dependencies
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
        parallel \
        python3 \
        python3-pip \
	; \
	rm -rf /var/lib/apt/lists/*

# Install Dependencies for FLAME GPU 2 (console, no python)
# There are not yet binary releases of the c++ static library which can be installed independently of models.
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		cmake \
        swig4.0 \
        git \
        python3 python3-pip python3-venv \
	; \
    python3 -m pip install wheel setuptools build matplotlib ; \
	rm -rf /var/lib/apt/lists/*; \
    gcc --version; \ 
    nvcc --version; \
    cmake --version

# Install Dependencies for Agents.jl
ENV JULIA_PATH /opt/julia
ENV PATH $JULIA_PATH/bin:$PATH
# @todo - check sha. Check gpg? 
RUN set -eux; \
    curl -fL -o julia.tar.gz "https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.2-linux-x86_64.tar.gz"; \
    mkdir -p ${JULIA_PATH}; \
    tar -xzf julia.tar.gz -C ${JULIA_PATH} --strip-components 1; \
    rm -rf julia.tar.gz; \
    julia --version

# Install Dependencies for NetLogo
# @todo - check sha? Though they are not provided...
ENV NETLOGO_PATH /opt/netlogo
ENV PATH $NETLOGO_PATH/bin:$PATH
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		default-jre \
	; \
	rm -rf /var/lib/apt/lists/*; \
    curl -fL -o netlogo.tgz "https://ccl.northwestern.edu/netlogo/6.3.0/NetLogo-6.3.0-64.tgz"; \
    mkdir -p ${NETLOGO_PATH}; \
    tar -xzf netlogo.tgz -C ${NETLOGO_PATH} --strip-components 1; \
    rm -rf netlogo.tgz; \
    JAVA_HOME=/usr ${NETLOGO_PATH}/netlogo-headless.sh --version

# # Install Dependecies for Mason 
# # https://cs.gmu.edu/~eclab/projbects/mason/mason.20.jar
# ENV MASON_PATH /opt/mason
# ENV PATH $MASON_PATH/bin:$PATH
# RUN set -eux; \
# 	apt-get update; \
# 	apt-get install -y --no-install-recommends \
# 		default-jre \
# 	; \
#     rm -rf /var/lib/apt/lists/*; \
#     curl -fL -o mason.jar "https://cs.gmu.edu/~eclab/projbects/mason/mason.20.jar"; \
#     mkdir -p ${MASON_PATH}; \
#     tar -xzf mason.jar -C ${MASON_PATH} --strip-components 1; \
#     rm -rf mason.jar; \


# Install Dependencies for Mesa
# @todo - add a user (or otherworkaround to avoid the pip running usera s root error.)
# @todo - pin versions for reproducibililty / move to a requirment.txt?
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		python3 python3-pip \
	; \
	python3 -m pip install mesa==1.0; \
    python3 -c "import mesa; print(mesa.__version__)"

