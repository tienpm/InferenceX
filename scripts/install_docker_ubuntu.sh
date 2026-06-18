#!/usr/bin/env bash
set -euo pipefail

if command -v docker >/dev/null 2>&1; then
    docker --version
    exit 0
fi

if [[ "$(id -u)" != "0" ]]; then
    echo "Run this script as root, or install Docker before running the benchmark." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl docker.io

if command -v systemctl >/dev/null 2>&1; then
    systemctl enable --now docker
else
    service docker start || true
fi

docker --version
docker info >/dev/null
