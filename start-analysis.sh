#!/usr/bin/env bash

source env/bin/activate

LOCAL_LOGS=../deepracer-local/data/robomaker/log
sudo mount --bind $LOCAL_LOGS logs

jupyter lab --no-browser 

sudo umount logs
