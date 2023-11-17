#!/bin/bash

# Check if a process ID was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <process-id>"
    exit 1
fi

# Collect performance data
perf stat -e uops_dispatched.port_0,uops_dispatched.port_1,uops_dispatched.port_2_3,uops_dispatched.port_4_9,uops_dispatched.port_5,uops_dispatched.port_6,uops_dispatched.port_7_8 -a -p "$1" -o E_output.csv -I 1 -x ","  sleep 5 
