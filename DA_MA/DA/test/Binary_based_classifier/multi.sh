#!/bin/bash


# array in shell script
arr=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18"  )

# @ means all elements in array
for i in ${arr[@]}; do
    sudo python3 run.py 10 &
done

