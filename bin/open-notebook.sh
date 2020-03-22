#!/bin/sh

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

URL=$($SCRIPTPATH/url-to-notebook.sh)

if test -z $URL; then
    echo "Could not find notebook details, have you run start.sh?"
else
    browse $URL
fi
