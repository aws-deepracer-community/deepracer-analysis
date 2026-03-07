#!/bin/bash
set -e

# Allow overriding the ubuntu user's UID/GID at runtime so that
# bind-mounted volumes have the correct ownership on the host.
#
# Usage:
#   docker run -e NB_UID=$(id -u) -e NB_GID=$(id -g) ...

NB_UID=${NB_UID:-1000}
NB_GID=${NB_GID:-1000}

CURRENT_UID=$(id -u ubuntu)
CURRENT_GID=$(id -g ubuntu)

if [ "${NB_GID}" != "${CURRENT_GID}" ]; then
    groupmod -g "${NB_GID}" ubuntu
fi

if [ "${NB_UID}" != "${CURRENT_UID}" ]; then
    usermod -u "${NB_UID}" ubuntu
    # Only chown when the UID actually changed — venv is world-readable so skip it
    chown -R ubuntu:ubuntu /workspace /home/ubuntu
fi

# Drop privileges and exec the given command (default: CMD)
exec gosu ubuntu "$@"
