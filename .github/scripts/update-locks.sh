#!/usr/bin/env bash
# Lock each module's environment.yaml into <module>/conda-lock.yml, the file the
# module's image is built from (conda-lock keeps the pip packages and pins git
# dependencies to their commit). Pass module names to lock only those.
# Requires conda-lock and micromamba on PATH. Run from the repo root.
set -euo pipefail

# model: the OGS and gmsh wheels have no linux-aarch64 build; communication_sdh: neither has vtk 9.4.2
platforms() {
  case "$1" in
    model|communication_sdh) echo "-p linux-64" ;;
    *)     echo "-p linux-64 -p linux-aarch64" ;;
  esac
}

modules=("$@")
if [ ${#modules[@]} -eq 0 ]; then
  for env in $(find . -mindepth 2 -maxdepth 2 -name environment.yaml | sort); do
    m=$(dirname "$env"); m=${m#./}
    [ "$m" = figure ] || modules+=("$m")
  done
fi

for m in "${modules[@]}"; do
  echo "[$m]"
  conda-lock lock -f "$m/environment.yaml" $(platforms "$m") --lockfile "$m/conda-lock.yml"
done
