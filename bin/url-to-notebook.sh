#!/bin/sh

URL="http://127.0.0.1:8888"

if test ! -z $1; then
    URL=$1
fi

TOKEN=$( docker logs deepracer-analysis 2>&1 | grep -o -E "token=[0-9a-f]+" | head -n 1)

if test ! -z "$TOKEN"; then
    echo $URL/?$TOKEN
fi
