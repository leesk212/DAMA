#!/bin/bash

mpstat 1 5 | awk -v OFS=',' 'NR>3 {print $4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14}' > U_output.csv
