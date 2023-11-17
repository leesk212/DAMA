#!/bin/bash

for i in {1..1200}; do
    echo "$i"
    current_time=$(date +"%Y-%m-%d_%H:%M:%S")

    mkdir "./result/$current_time"
    cd "./result/$current_time"

    ../../monitoring_tool/U_.sh &
    #../../monitoring_tool/U.sh &
    ../../monitoring_tool/S.sh &

    sleep 7

    #kill -9 $(pgrep tshark)
    cd ../../
done

