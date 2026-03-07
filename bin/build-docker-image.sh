#!/bin/sh
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$SCRIPTPATH/.."

docker build -f docker/Dockerfile -t deepracer-analysis .
