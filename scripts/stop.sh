#!/usr/bin/env bash
set -euo pipefail

# Stop and remove the entire Docker Compose stack
docker compose down
echo "Docker Compose services stopped."
