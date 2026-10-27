# The image of one module, built from its conda-lock.yml (conda and pip packages
# at the locked versions). Build from the repo root:
#   docker build -f .github/templates/Dockerfile --build-arg MODULE=sampling -t sampling .
FROM mambaorg/micromamba:2.3.1
ARG MODULE
ARG MAMBA_DOCKERFILE_ACTIVATE=1
USER root
COPY ${MODULE}/conda-lock.yml /tmp/conda-lock.yml
# conda-lock (with git, for the pip packages from git) is only used for the install
RUN micromamba install -y -n base -c conda-forge conda-lock git \
    && conda-lock install -p /opt/env /tmp/conda-lock.yml \
    && micromamba remove -y -n base conda-lock git \
    && micromamba clean --all --yes
# ps, which Nextflow uses for the task metrics
RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER
ENV PATH=/opt/env/bin:$PATH
