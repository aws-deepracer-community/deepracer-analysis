#!/bin/sh

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

if test "$1" = "lab"; then
    LAB_ENV="-e JUPYTER_ENABLE_LAB=yes"
fi

docker run --rm -d -p 8888:8888 $LAB_ENV \
-v "$SCRIPTPATH/..":/home/jovyan/work \
-v ~/.aws:/home/jovyan/.aws \
--name deepracer-analysis deepracer-analysis
