# The image of surrogate: the packages of conda-lock.yml. Build with the module folder as context:
#   docker build -t surrogate surrogate/
FROM mambaorg/micromamba:2.3.1
ARG MAMBA_DOCKERFILE_ACTIVATE=1
USER root
COPY conda-lock.yml /tmp/conda-lock.yml
# git, gcc and conda-lock are only needed for the install and are removed again
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        # required by nextflow: ps for the task metrics
        procps \
        # required by pip packages from git repositories
        git ca-certificates \
        # required for compilation of C source
        gcc libc6-dev \
    && micromamba install -y -n base -c conda-forge conda-lock \
    && conda-lock install -p /opt/env /tmp/conda-lock.yml \
    && micromamba remove -y -n base conda-lock \
    && micromamba clean --all --yes \
    && apt-get purge -y --auto-remove git gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER
ENV PATH=/opt/env/bin:$PATH
