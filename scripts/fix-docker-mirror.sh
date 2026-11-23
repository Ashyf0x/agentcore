#!/usr/bin/env bash
# Point the Docker daemon at a reachable registry mirror and drop the stale
# localhost proxy. SWE-bench needs one image per task, pulled from Docker Hub,
# and the daemon's current proxy (127.0.0.1:7890) no longer exists in WSL.
#
# Usage: sudo bash scripts/fix-docker-mirror.sh
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "run with sudo: sudo bash $0" >&2
    exit 1
fi

if [ -f /etc/docker/daemon.json ]; then
    backup="/etc/docker/daemon.json.bak.$(date +%Y%m%d%H%M%S)"
    cp /etc/docker/daemon.json "$backup"
    echo "backed up old config to $backup"
fi

cat > /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": ["https://docker.1ms.run"]
}
EOF

systemctl restart docker
sleep 3
docker info --format "mirrors={{json .RegistryConfig.Mirrors}}"
echo "done. verify with: docker pull hello-world"