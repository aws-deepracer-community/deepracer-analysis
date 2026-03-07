#!/bin/sh
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

docker run --rm -d \
  -p 127.0.0.1:8888:8888 \
  -e NB_UID="$(id -u)" \
  -e NB_GID="$(id -g)" \
  -v "$SCRIPTPATH/../analysis:/workspace/analysis" \
  -v ~/.aws:/home/ubuntu/.aws \
  --name deepracer-analysis deepracer-analysis
