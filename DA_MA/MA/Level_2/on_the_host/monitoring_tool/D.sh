#!/bin/bash

sudo tcpdump -i tap0 udp -c 3  > output.csv
