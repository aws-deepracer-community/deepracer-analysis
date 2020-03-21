#!/bin/sh

PWD=$(pwd)

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cd $SCRIPTPATH/..

mkdir docker-build

cp docker/* docker-build
cp requirements.txt docker-build

cd docker-build

docker build . -t deepracer-analysis

cd ..

rm -rf docker-build

cd $PWD
