#!/bin/bash


tshark -i tap0 -T fields -E header=y -E separator=, -e frame.time_delta -e frame.len -e ip.proto -e ip.len -e ip.ttl -e tcp.len -e tcp.window_size -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -c 5000 > N_output.csv
