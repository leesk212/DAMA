#!/bin/bash

for i in {1..84}; do
    current_time=$(date +"%Y-%m-%d_%H:%M:%S")

    mkdir "./result/$current_time"
    cd "./result/$current_time"

    ../../monitoring_tool/E.sh &
    ../../monitoring_tool/U.sh &
    ../../monitoring_tool/N.sh &

    sleep 7

    kill -9 $(pgrep tshark)
    cd ../../
done

